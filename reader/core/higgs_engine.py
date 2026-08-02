"""Higgs TTS 3 engine using the Transformers-compatible community port.

The official Boson model card is the authority for capabilities and control
tokens.  Direct local inference follows the adapter in
``source/higgs-tts-3-4b/app.py`` because the official weight repository does
not currently expose the custom Transformers ``auto_map`` implementation.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import itertools
import logging
import math
import os
import re
import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from core import tts_engine as _omnivoice
from core.pronunciation import apply_pronunciation, lexicon_version
from core.tts_engine import (
    AUDIO_CACHE_DIR,
    SAMPLE_RATE,
    _audio_zcr,
    _voice_design_ref_language,
    _voice_design_ref_text,
    _voice_lock_enabled,
    _write_audio_atomic,
    apply_text_normalization,
)

log = logging.getLogger(__name__)

OFFICIAL_MODEL_REPO = "bosonai/higgs-tts-3-4b"
DEFAULT_TRANSFORMERS_REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
REFERENCE_EXPAND_IF_SHORTER_SECONDS = 2.0
REFERENCE_EXPAND_TARGET_SECONDS = 4.0
HIGGS_CACHE_VERSION = 6
HIGGS_MODEL_INIT_SEED = 123
# Higgs picks the speaker while it decodes, so with the default random seed
# every segment is read by a different voice. The locked narrator clip is
# rendered once with this seed and cloned for every segment afterwards.
HIGGS_VOICE_LOCK_SEED = 20260101
QUANTIZATION_MODES = ("none", "8bit", "4bit")
MAX_CONCURRENCY = 4
# Warn below this share of total VRAM. Above it the driver starts paging
# weights through host memory, which costs a PCIe transfer per decoded token.
VRAM_PRESSURE_RATIO = 0.9

_OMNIVOICE_TAGS = {
    "laughter": "<|sfx:laughter|>Haha",
    "sigh": "<|sfx:sigh|>Uh",
    "dissatisfaction-hnn": "<|emotion:bitterness|>",
    "confirmation-en": "",
    "question-ei": "<|emotion:contemplation|>",
    "question-oh": "<|emotion:surprise|>",
    "question-ah": "<|emotion:confusion|>",
    "surprise-ah": "<|emotion:surprise|>",
    "surprise-oh": "<|emotion:surprise|>",
    "surprise-wa": "<|emotion:awe|>",
    "surprise-yo": "<|emotion:elation|>",
}
_BRACKET_TAG_RE = re.compile(r"\[([a-z0-9_-]+)\]", re.IGNORECASE)
_WORKER_MARKER = "AURIS_HIGGS_JSON:"
_JSON_DECODER = json.JSONDecoder()


def _setting(key: str, default: Any) -> Any:
    try:
        from core.settings import get

        return get(key, default)
    except Exception:
        return default


def _prepare_reference(audio: Any, sample_rate: int) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(f"Reference audio must be mono or stereo, got shape {array.shape}")
    seconds = len(array) / float(sample_rate) if sample_rate else 0.0
    if 0 < seconds < REFERENCE_EXPAND_IF_SHORTER_SECONDS:
        target = int(REFERENCE_EXPAND_TARGET_SECONDS * sample_rate)
        repeats = max(1, int(math.ceil(target / max(len(array), 1))))
        array = np.tile(array, repeats)
    return array


def _speed_token(speed: float) -> str:
    if speed <= 0.72:
        return "<|prosody:speed_very_slow|>"
    if speed < 0.94:
        return "<|prosody:speed_slow|>"
    if speed >= 1.32:
        return "<|prosody:speed_very_fast|>"
    if speed > 1.08:
        return "<|prosody:speed_fast|>"
    return ""


def _instruct_tokens(instruct: str | None) -> str:
    value = str(instruct or "").lower()
    tokens: list[str] = []
    if "whisper" in value:
        tokens.append("<|style:whispering|>")
    if "very low pitch" in value or "low pitch" in value:
        tokens.append("<|prosody:pitch_low|>")
    elif "very high pitch" in value or "high pitch" in value:
        tokens.append("<|prosody:pitch_high|>")
    return "".join(tokens)


def _translate_inline_tags(text: str) -> str:
    return _BRACKET_TAG_RE.sub(
        lambda match: _OMNIVOICE_TAGS.get(match.group(1).lower(), ""),
        text,
    )


def _language_cleanup(text: str, language: str | None) -> str:
    """Repair common legacy-PDF glyph substitutions before Higgs tokenization."""
    code = str(language or "").strip().lower()
    if code in {"hu", "hun", "hungarian", "magyar"}:
        return text.translate(str.maketrans({"õ": "ő", "Õ": "Ő", "û": "ű", "Û": "Ű"}))
    return text


def _prefix_control(category: str, value: str) -> str:
    value = str(value or "").strip()
    if not value or value == "none":
        return ""
    return f"<|{category}:{value}|>"


def _parse_worker_response_line(line: str) -> dict | None:
    """Extract one framed JSON reply from stdout mixed with progress output."""
    marker_at = line.find(_WORKER_MARKER)
    if marker_at < 0:
        return None
    payload = line[marker_at + len(_WORKER_MARKER):].lstrip()
    response, _ = _JSON_DECODER.raw_decode(payload)
    if not isinstance(response, dict):
        raise RuntimeError("Higgs worker returned a non-object response")
    return response


class HiggsTTSEngine:
    engine_name = "higgs"
    # Higgs picks the speaker while it decodes, so it needs the same pinning.
    uses_voice_lock = True

    def __init__(self, model_path: str = "", worker_label: str = "primary"):
        self.model_path = model_path
        self.worker_label = worker_label
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self._generating = threading.Event()
        self._loading = False
        self._ready = False
        self._cancel_load = threading.Event()
        self._error: str | None = None
        self._resolved_model = ""
        self._worker: subprocess.Popen | None = None
        self._sample_rate = SAMPLE_RATE
        self._load_metadata: dict = {}
        # Reference clips the running worker has already encoded. Cleared with
        # the worker, because the codes live in that process.
        self._worker_reference_keys: set[str] = set()
        # In-flight RPCs, keyed by request id. A single reader thread owns the
        # worker's stdout and wakes the caller each reply belongs to.
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._request_ids = itertools.count(1)
        self._generating_count = 0
        self._generating_lock = threading.Lock()
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    def _source(self) -> tuple[str, bool]:
        source = str(_setting("higgs_model_source", "download")).lower()
        path = self.model_path or str(_setting("higgs_model_path", "") or "")
        if source == "local":
            return path, True
        repo = str(_setting("higgs_model_repo", DEFAULT_TRANSFORMERS_REPO) or DEFAULT_TRANSFORMERS_REPO)
        if repo in {OFFICIAL_MODEL_REPO, "bosonai/higgs-audio-v3-tts-4b"}:
            repo = DEFAULT_TRANSFORMERS_REPO
        return repo, False

    @staticmethod
    def _quantization() -> str:
        mode = str(_setting("higgs_quantization", "none") or "none").lower()
        return mode if mode in QUANTIZATION_MODES else "none"

    @classmethod
    def _concurrency(cls) -> int:
        """How many segments the worker may decode at once.

        Memory-wise this is nearly free — the threads share the weights and the
        codec, and only the KV cache is per request. It still makes things
        worse on hardware where one decode stream already saturates the GPU.
        Measured on an RTX 5070 Laptop at 4-bit, with peak use at 4.7 of 7.96
        GiB so nothing was paging: 1 lane 0.71x realtime, 2 lanes 0.43x, 3
        lanes 0.31x. NF4 dequantization makes every matmul real work, so there
        is no idle time to fill, and the decode loops then contend for the GIL.
        Auto therefore stays serial; raise it only if your card shows otherwise.
        """
        try:
            requested = int(_setting("higgs_concurrency", 0) or 0)
        except (TypeError, ValueError):
            requested = 0
        if requested > 0:
            return max(1, min(requested, MAX_CONCURRENCY))
        return 1

    def _accel_status(self) -> dict:
        """Report what the worker actually loaded, not what we hoped for."""
        meta = self._load_metadata
        device = str(meta.get("device") or "?")
        dtype = str(meta.get("dtype") or "?").replace("torch.", "")
        quantization = str(meta.get("quantization") or "none")
        precision = dtype if quantization == "none" else f"{quantization} ({dtype} compute)"
        message = f"Higgs generate_speech ({precision} on {device})"
        allocated = meta.get("vram_allocated_gib")
        total = meta.get("vram_total_gib")
        if allocated is not None and total:
            message += f" · {allocated:.2f}/{total:.2f} GiB VRAM"
            if allocated > total * VRAM_PRESSURE_RATIO:
                message += " · weights exceed VRAM, paging over PCIe"
        concurrency = int(meta.get("max_concurrency", 1) or 1)
        if concurrency > 1:
            message += f" · {concurrency} concurrent segments"
        return {
            "effective": "transformers",
            "message": message,
            "device": device,
            "dtype": dtype,
            "quantization": quantization,
            "concurrency": concurrency,
            "vram_allocated_gib": allocated,
            "vram_total_gib": total,
        }

    def status(self) -> dict:
        source, local_only = self._source()
        base = {"engine": self.engine_name, "model": self._resolved_model or source}
        if self._error:
            return {**base, "state": "error", "message": self._error}
        if self._ready:
            return {
                **base,
                "state": "ready",
                "generating": self._generating.is_set(),
                "accel": self._accel_status(),
            }
        if self._loading:
            return {**base, "state": "loading"}
        return {
            **base,
            "state": "not_loaded",
            "model_path": source,
            "model_exists": (os.path.isdir(source) if local_only else True),
        }

    def load_async(self) -> None:
        if self._ready or self._loading:
            return
        self._cancel_load.clear()
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self) -> None:
        self._cancel_load.clear()
        self._load()
        if not self._ready:
            raise RuntimeError(self._error or "Higgs TTS model failed to load")

    def _load(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._loading = True
            self._error = None
        try:
            source, local_only = self._source()
            if local_only and not os.path.isdir(source):
                raise FileNotFoundError(
                    f"Higgs model not found at: {source}. Configure its own path in Settings."
                )
            runtime = Path(__file__).resolve().parent.parent / ".higgs_runtime"
            if not (runtime / "transformers").is_dir():
                raise RuntimeError(
                    "The isolated Higgs Transformers runtime is missing. Run "
                    r"reader\.venv\Scripts\python.exe -m pip install --target "
                    r"reader\.higgs_runtime --no-deps transformers==5.13.0"
                )
            env = os.environ.copy()
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(runtime) + (os.pathsep + existing if existing else "")
            worker_path = Path(__file__).with_name("higgs_worker.py")
            self._worker = subprocess.Popen(
                [sys.executable, "-u", str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                env=env,
            )
            self._worker_reference_keys.clear()
            self._start_reply_reader(self._worker)
            response = self._rpc_raw(
                {
                    "source": source,
                    "local_only": local_only,
                    "model_seed": HIGGS_MODEL_INIT_SEED,
                    "quantization": self._quantization(),
                    "max_concurrency": self._concurrency(),
                }
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Higgs worker failed to start")
            if self._cancel_load.is_set():
                log.info("Higgs load cancelled for import-time LLM analysis.")
                self.unload()
                return
            self._sample_rate = int(response.get("sample_rate", SAMPLE_RATE))
            self._load_metadata = dict(response)
            self._resolved_model = source
            if self._cancel_load.is_set():
                self.unload()
                return
            self._ready = True
            self.model = self._worker  # resident-worker marker used by lifecycle code
            log.info(
                "Higgs TTS ready (%s, Transformers %s, %s %s, quantization=%s, "
                "model seed=%s, audio head shared=%s).",
                source,
                response.get("transformers", "?"),
                response.get("dtype", "?"),
                response.get("device", "?"),
                response.get("quantization", "none"),
                response.get("model_seed", "?"),
                response.get("audio_head_shared", False),
            )
            self._warn_on_vram_pressure(response)
        except Exception as exc:
            self._error = str(exc)
            self.model = None
            self.tokenizer = None
            if self._worker is not None:
                self._worker.terminate()
                self._worker = None
            log.error("Failed to load Higgs TTS: %s", exc)
        finally:
            self._loading = False

    @staticmethod
    def _warn_on_vram_pressure(response: dict) -> None:
        """Name the real cost when the weights do not fit on the card.

        Windows and WSL page the overflow into host memory instead of raising
        OOM, so the only symptom is that every decoded audio token waits on a
        PCIe transfer. Without this line the slowdown looks like the model
        simply being slow.
        """
        allocated = response.get("vram_allocated_gib")
        total = response.get("vram_total_gib")
        if not allocated or not total or allocated <= total * VRAM_PRESSURE_RATIO:
            return
        log.warning(
            "Higgs weights need %.2f GiB but the GPU has %.2f GiB. The driver "
            "will page them through host memory and generation will be very "
            "slow. Set Higgs quantization to 4-bit or 8-bit in Settings.",
            allocated,
            total,
        )

    def unload(self) -> None:
        self._cancel_load.set()
        self._worker_reference_keys.clear()
        worker = self._worker
        if worker is not None:
            try:
                if worker.poll() is None:
                    self._rpc_raw({"command": "shutdown"})
                    worker.wait(timeout=5)
            except Exception:
                # _rpc_raw() or a concurrent load/cancel path may already have
                # cleared self._worker. Always clean up the captured process.
                if worker.poll() is None:
                    worker.terminate()
            if self._worker is worker:
                self._worker = None
        self.model = None
        self.tokenizer = None
        self._ready = False
        # If startup is in flight, _load owns the transition back to false.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def reload(self) -> None:
        self.unload()
        self._error = None
        self.load_async()

    def set_dedicated_cuda_stream(self, enabled: bool) -> None:
        # The community Transformers port exposes only whole-waveform generation.
        return

    def cancel(self) -> bool:
        """Stop the active worker process and reload the cached model."""
        worker = self._worker
        if (
            worker is None
            or worker.poll() is not None
            or not self._generating.is_set()
        ):
            return False
        try:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            if self._worker is worker:
                self._worker = None
                self.model = None
                self.tokenizer = None
                self._ready = False
                self._loading = False
                self._error = None
                self._worker_reference_keys.clear()
            # The weights are already in the HF cache. Reload asynchronously so
            # the next Play can resume without restarting the whole app.
            self.load_async()
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def invalidate_voice_prompt(self, ref_audio=None, ref_text=None) -> None:
        # Higgs conditions directly on the reference waveform for each call.
        return

    def _get_voice_clone_prompt(self, ref_audio, ref_text):
        return None

    @staticmethod
    def _generation_settings() -> dict:
        top_p = float(_setting("higgs_top_p", 0.95))
        top_k = int(_setting("higgs_top_k", 50))
        return {
            "temperature": float(_setting("higgs_temperature", 0.8)),
            "top_p": top_p if top_p > 0 else None,
            "top_k": top_k if top_k > 0 else None,
            "max_new_tokens": int(_setting("higgs_max_new_tokens", 1024)),
            "seed": int(_setting("higgs_seed", -1)),
        }

    @classmethod
    def cache_key(
        cls,
        text: str,
        instruct: str | None,
        ref_audio: str | None,
        speed: float,
        ref_text: str | None = None,
        language: str | None = None,
        normalize_text: bool = False,
        num_step: int = 0,
        lexicon: str | None = None,
    ) -> str:
        controls = (
            _setting("higgs_prompt_mode", "raw"),
            _setting("higgs_default_emotion", "none"),
            _setting("higgs_default_style", "none"),
            _setting("higgs_default_expressive", "none"),
        )
        # Quantized weights produce different audio, so they need their own
        # cache entries. Full precision stays out of the payload to keep the
        # keys of everything generated before quantization existed — on this
        # engine a re-render of a whole book is expensive.
        quantization = cls._quantization()
        quantization_tag = "" if quantization == "none" else f"|q={quantization}"
        generation = cls._generation_settings()
        payload = (
            f"higgs-v{HIGGS_CACHE_VERSION}|{text}|{instruct}|{ref_audio}|{ref_text}|{speed:.3f}|"
            f"{language or ''}|nt={int(bool(normalize_text))}|{controls}|{generation}|"
            f"lex={lexicon_version(lexicon)}{quantization_tag}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def cache_path(key: str) -> str:
        return os.path.join(AUDIO_CACHE_DIR, f"{key}.wav")

    def _prompt(
        self,
        text: str,
        instruct: str | None,
        speed: float,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None = None,
    ) -> str:
        text = apply_pronunciation(_language_cleanup(text, language), lexicon)
        prompt_mode = str(_setting("higgs_prompt_mode", "raw") or "raw").lower()
        if prompt_mode == "raw":
            # Match source/higgs-tts-3-4b/app.py's default compose_prompt path:
            # clean user text, no implicit normalization or delivery prefix.
            # Auris enrichment tags are implementation details of OmniVoice
            # and must not reach Higgs as literal bracketed words.
            return _BRACKET_TAG_RE.sub("", text).strip()

        spoken = apply_text_normalization(text, language) if normalize_text else text
        spoken = _translate_inline_tags(spoken).strip()
        has_emotion = spoken.startswith("<|emotion:")
        has_style = spoken.startswith("<|style:") or "<|style:" in spoken[:100]
        has_expressive = "<|prosody:expressive_" in spoken[:160]
        prefix = "".join(
            [
                "" if has_emotion else _prefix_control(
                    "emotion", _setting("higgs_default_emotion", "none")
                ),
                "" if has_style else _prefix_control(
                    "style", _setting("higgs_default_style", "none")
                ),
                "" if has_expressive else _prefix_control(
                    "prosody", _setting("higgs_default_expressive", "none")
                ),
                _instruct_tokens(instruct),
                _speed_token(speed),
            ]
        )
        return f"{prefix}{spoken}"

    def _start_reply_reader(self, worker: subprocess.Popen) -> None:
        thread = threading.Thread(
            target=self._read_replies, args=(worker,), daemon=True,
            name="higgs-replies",
        )
        thread.start()

    def _read_replies(self, worker: subprocess.Popen) -> None:
        """Route worker replies to their waiting callers.

        One reader owns stdout so several segments can be in flight at once;
        each reply carries the id of its request.
        """
        try:
            for line in worker.stdout:
                try:
                    response = _parse_worker_response_line(line)
                except (ValueError, RuntimeError) as exc:
                    log.warning("Unparsable Higgs worker reply (%s): %s", exc, line.rstrip())
                    continue
                if response is None:
                    log.info("Higgs worker: %s", line.rstrip())
                    continue
                self._deliver(response)
        except (OSError, ValueError):
            pass
        finally:
            self._fail_pending(
                RuntimeError(f"Higgs worker exited unexpectedly (code {worker.poll()})")
            )

    def _deliver(self, response: dict) -> None:
        request_id = response.get("id")
        with self._pending_lock:
            if request_id is None:
                # A startup failure is reported before the worker has parsed
                # the init request, so it arrives without an id. Hand it to the
                # oldest caller, which is the one still waiting for startup.
                if not self._pending:
                    log.info("Higgs worker reply with no waiter: %s", response)
                    return
                request_id = min(self._pending)
            pending = self._pending.pop(request_id, None)
        if pending is None:
            log.info("Higgs worker reply for unknown request %s", request_id)
            return
        pending["response"] = response
        pending["event"].set()

    def _fail_pending(self, error: Exception) -> None:
        with self._pending_lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for pending in waiting:
            pending["error"] = error
            pending["event"].set()

    def _rpc_raw(self, payload: dict) -> dict:
        worker = self._worker
        if worker is None or worker.stdin is None or worker.stdout is None:
            raise RuntimeError("Higgs worker is not running")
        request_id = next(self._request_ids)
        pending = {"event": threading.Event(), "response": None, "error": None}
        with self._pending_lock:
            self._pending[request_id] = pending
        try:
            with self._stdin_lock:
                worker.stdin.write(
                    json.dumps({**payload, "id": request_id}, ensure_ascii=True) + "\n"
                )
                worker.stdin.flush()
        except (OSError, ValueError) as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise RuntimeError(f"Higgs worker is not accepting requests: {exc}") from exc
        pending["event"].wait()
        if pending["error"] is not None:
            raise pending["error"]
        return pending["response"]

    @contextlib.contextmanager
    def _generation_in_flight(self):
        """Track concurrent generations so status() stays truthful."""
        with self._generating_lock:
            self._generating_count += 1
            self._generating.set()
        try:
            yield
        finally:
            with self._generating_lock:
                self._generating_count = max(0, self._generating_count - 1)
                if self._generating_count == 0:
                    self._generating.clear()

    @staticmethod
    def _reference_key(ref_audio: str) -> str:
        """Content identity of a reference clip, stable across temp copies."""
        stat = os.stat(ref_audio)
        payload = (
            f"{os.path.abspath(ref_audio)}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{REFERENCE_EXPAND_IF_SHORTER_SECONDS}|{REFERENCE_EXPAND_TARGET_SECONDS}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _prepare_reference_file(ref_audio: str) -> tuple[str, str | None]:
        """Return the path to send and the temp file to delete afterwards."""
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        audio, sr = sf.read(ref_audio, always_2d=False)
        processed = _prepare_reference(audio, int(sr))
        if len(processed) == len(np.asarray(audio).squeeze()):
            return ref_audio, None
        handle, temp_path = tempfile.mkstemp(suffix=".wav", prefix="auris-higgs-ref-")
        os.close(handle)
        sf.write(temp_path, processed, int(sr))
        return temp_path, temp_path

    def _generate_rpc(
        self, request: dict, ref_audio: str | None, attach_reference: bool
    ) -> dict:
        """One generate call, attaching the reference clip only when needed."""
        temp_path = None
        payload = dict(request)
        if ref_audio and attach_reference:
            send_path, temp_path = self._prepare_reference_file(ref_audio)
            payload["reference_audio"] = send_path
        try:
            # Deliberately unlocked: the worker multiplexes on request id, so
            # several segments decode at once.
            return self._rpc_raw(payload)
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _synthesize(
        self,
        text: str,
        instruct: str | None,
        ref_audio: str | None,
        ref_text: str | None,
        speed: float,
        language: str | None,
        normalize_text: bool,
        seed: int | None = None,
        lexicon: str | None = None,
    ) -> np.ndarray:
        if not self._ready or self._worker is None:
            raise RuntimeError("Higgs TTS is not loaded. " + (self._error or "Load it first."))
        settings = self._generation_settings()
        configured_seed = settings.pop("seed")
        seed = configured_seed if seed is None else int(seed)
        reference_key = self._reference_key(ref_audio) if ref_audio else None
        prompt = self._prompt(text, instruct, speed, language, normalize_text, lexicon)
        handle, output_path = tempfile.mkstemp(suffix=".wav", prefix="auris-higgs-out-")
        os.close(handle)
        request = {
            "command": "generate",
            "prompt": prompt,
            "generation": settings,
            "seed": seed,
            "reference_audio": None,
            "reference_key": reference_key,
            "reference_text": str(ref_text or "").strip() or None,
            "output_path": output_path,
        }
        try:
            with self._generation_in_flight():
                # The worker keeps the encoded reference codes, so a locked
                # narrator clip is encoded once per book instead of once per
                # sentence. An evicted entry is reported, not guessed at.
                cached = bool(reference_key) and reference_key in self._worker_reference_keys
                response = self._generate_rpc(request, ref_audio, not cached)
                if response.get("error_code") == "reference_cache_miss":
                    self._worker_reference_keys.discard(reference_key)
                    response = self._generate_rpc(request, ref_audio, True)
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Higgs generation failed")
            if reference_key and response.get("reference_cached"):
                self._worker_reference_keys.add(reference_key)
            audio, _ = sf.read(output_path, dtype="float32")
            return np.asarray(audio, dtype=np.float32)
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass

    def _voice_identity(self, instruct: str | None, language: str | None) -> str:
        return (
            f"higgs|{instruct or ''}|{_voice_design_ref_language(language)}|"
            f"{self._resolved_model or ''}|{_setting('higgs_prompt_mode', 'raw')}"
        )

    def _voice_ref_path(self, instruct: str | None, language: str | None) -> str:
        digest = hashlib.md5(
            self._voice_identity(instruct, language).encode("utf-8")
        ).hexdigest()
        return os.path.join(_omnivoice.VOICE_REF_DIR, f"higgs_{digest}.wav")

    def _voice_lock_seed(self, instruct: str | None, language: str | None) -> int:
        """Seed for the narrator clip, derived from the narrator description.

        In ``raw`` prompt mode the instruct never reaches the model, so a
        constant seed would give every book the same speaker with no way to
        change it. Deriving the seed from the description keeps the voice
        reproducible while letting an edited instruct pick a different one.
        """
        digest = hashlib.md5(
            self._voice_identity(instruct, language).encode("utf-8")
        ).hexdigest()
        return (HIGGS_VOICE_LOCK_SEED + int(digest[:8], 16)) % 2_147_483_647

    def _ensure_locked_voice(
        self, instruct: str | None, language: str | None
    ) -> tuple[str, str]:
        """Render the narrator clip once, then reuse it as the clone source."""
        ref_path = self._voice_ref_path(instruct, language)
        ref_text = _voice_design_ref_text(language)
        if os.path.exists(ref_path):
            return ref_path, ref_text

        log.info("Rendering locked Higgs narrator voice (%s)", language or "default")
        audio = self._synthesize(
            ref_text,
            instruct,
            None,
            None,
            1.0,
            language,
            False,
            seed=self._voice_lock_seed(instruct, language),
        )
        if len(audio) == 0 or _audio_zcr(audio) <= 0:
            raise RuntimeError("Higgs returned no usable audio for the narrator clip")

        os.makedirs(_omnivoice.VOICE_REF_DIR, exist_ok=True)
        _write_audio_atomic(ref_path, audio, self._sample_rate)
        return ref_path, ref_text

    def _resolve_voice(
        self,
        instruct: str | None,
        ref_audio: str | None,
        ref_text: str | None,
        language: str | None,
    ) -> tuple[str | None, str | None]:
        """Pin the narrator to one voice unless a real reference was supplied."""
        if ref_audio or not _voice_lock_enabled():
            return ref_audio, ref_text
        try:
            return self._ensure_locked_voice(instruct, language)
        except Exception as exc:
            log.warning(
                "Higgs voice lock unavailable (%s); each segment keeps its own voice.",
                exc,
            )
            return ref_audio, ref_text

    def generate(
        self,
        text: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        speed: float = 1.0,
        num_step: int | None = None,
        language: str | None = None,
        normalize_text: bool | None = None,
        lexicon: str | None = None,
    ) -> dict:
        if normalize_text is None:
            normalize_text = bool(_setting("normalize_text", True))
        ref_audio, ref_text = self._resolve_voice(instruct, ref_audio, ref_text, language)
        key = self.cache_key(
            text,
            instruct,
            ref_audio,
            speed,
            ref_text=ref_text,
            language=language,
            normalize_text=bool(normalize_text),
            lexicon=lexicon,
        )
        path = self.cache_path(key)
        if os.path.exists(path):
            data, sr = sf.read(path)
            return {
                "audio_path": path,
                "duration_sec": len(data) / sr,
                "cache_hit": True,
                "cache_key": key,
            }
        audio = self._synthesize(
            text, instruct, ref_audio, ref_text, speed, language, bool(normalize_text),
            lexicon=lexicon,
        )
        sample_rate = self._sample_rate
        _write_audio_atomic(path, audio, sample_rate)
        return {
            "audio_path": path,
            "duration_sec": len(audio) / sample_rate,
            "cache_hit": False,
            "cache_key": key,
        }

    def _generate_item(self, item: dict) -> dict:
        return self.generate(
            text=item["text"],
            instruct=item.get("instruct"),
            ref_audio=item.get("ref_audio"),
            ref_text=item.get("ref_text"),
            speed=float(item.get("speed") or 1.0),
            language=item.get("language"),
            normalize_text=item.get("normalize_text"),
            lexicon=item.get("lexicon"),
        )

    def generate_many(
        self,
        items: list[dict],
        num_step: int | None = None,
        batch_size: int | None = None,
        on_item=None,
        on_status=None,
    ) -> list[dict]:
        total = len(items)
        lanes = min(self._concurrency(), total)
        if lanes <= 1:
            results: list[dict] = []
            for index, item in enumerate(items):
                if on_status is not None:
                    on_status(f"Higgs utterance {index + 1}/{total}…")
                result = self._generate_item(item)
                results.append(result)
                if on_item is not None:
                    on_item(index, result)
            return results

        # The decode loop is latency bound — one small kernel per audio token
        # with Python between them — so overlapping segments keeps the GPU
        # busy during another lane's host-side work.
        outputs: list[dict | None] = [None] * total
        progress = {"done": 0}
        progress_lock = threading.Lock()

        def run(index: int) -> None:
            result = self._generate_item(items[index])
            outputs[index] = result
            with progress_lock:
                progress["done"] += 1
                done = progress["done"]
            if on_status is not None:
                on_status(f"Higgs utterance {done}/{total} · {lanes} lanes…")
            if on_item is not None:
                on_item(index, result)

        with ThreadPoolExecutor(
            max_workers=lanes, thread_name_prefix="higgs-lane"
        ) as executor:
            futures = [executor.submit(run, index) for index in range(total)]
            for future in futures:
                future.result()

        missing = [index for index, result in enumerate(outputs) if result is None]
        if missing:
            raise RuntimeError(f"Higgs left {len(missing)} segments unresolved")
        return outputs  # type: ignore[return-value]

    def generate_preview(
        self,
        instruct: str,
        sample_text: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        language: str | None = None,
        normalize_text: bool | None = None,
        speaker: str | None = None,
    ) -> dict:
        # speaker is part of the shared signature for Piper's per-character
        # casting; Higgs takes its voice from the instruct and reference clip.
        return self.generate(
            sample_text,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            normalize_text=normalize_text,
        )
