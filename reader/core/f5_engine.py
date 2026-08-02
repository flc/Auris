"""F5-TTS engine for the Hungarian community checkpoints.

Unlike Higgs, F5-TTS runs in-process.  Its ``transformers`` requirement is
unpinned and it never imports the pieces OmniVoice's 5.3.0 pin conflicts with,
so this engine needs neither an isolated runtime directory nor a worker
subprocess.

Two limitations shape the design and are deliberately surfaced rather than
worked around:

* F5-TTS has no voice-design path.  It can only clone a reference clip, so a
  narrator reference WAV plus its exact transcript must be configured before
  the engine can synthesize anything.  Characters without their own reference
  fall back to the narrator clip and therefore share its timbre.
* The Hungarian checkpoints ship a 67-token lowercase vocabulary and F5-TTS
  maps every unknown character to index 0 — a space.  Text has to be folded
  into that vocabulary before tokenization or words silently lose letters.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import logging
import os
import re
import threading
from typing import Any

import numpy as np
import soundfile as sf

from core.pronunciation import apply_pronunciation, lexicon_version
from core.tts_engine import (
    AUDIO_CACHE_DIR,
    _write_audio_atomic,
    apply_text_normalization,
    variant_cache_tag,
    variant_seed,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "Maxdorger29/f5-tts-hungarian"
DEFAULT_MODEL_FILE = "model_last_final.safetensors"
DEFAULT_VOCAB_FILE = "vocab.txt"
F5_CACHE_VERSION = 1
# The checkpoints are vocos-mel finetunes of SWivid/F5-TTS F5TTS_v1_Base and
# keep its architecture; only the vocabulary differs.
F5_SAMPLE_RATE = 24_000
F5_V1_BASE_ARCH = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "text_mask_padding": True,
    "qk_norm": None,
    "conv_layers": 4,
    "pe_attn_head": None,
}
# The DiT backbone emits a short burst while attention warms up. The upstream
# model card ships a fixed 350 ms cut; this energy-based variant only removes
# the burst itself so short utterances do not lose their first phoneme.
ONSET_TRIM_MAX_MS = 400
ONSET_TRIM_WINDOW_MS = 10
ONSET_TRIM_THRESHOLD_RATIO = 0.15
REFERENCE_MIN_SECONDS = 3.0

_BRACKET_TAG_RE = re.compile(r"\[([a-z0-9_-]+)\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\.{3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?:;…])")
# Folding parentheses onto commas doubles them up next to existing punctuation
# ("(mérve)," → ",mérve,,"), which the model reads as a stutter of pauses. A run
# collapses to its strongest mark so "2+2? (Semmi.)" keeps the question rather
# than the comma the closing parenthesis became.
_PUNCT_RUN_RE = re.compile(r"[,.!?:;…](?:\s*[,.!?:;…])+")
_PUNCT_RANK = {",": 0, ":": 1, ";": 2, ".": 3, "!": 4, "…": 5, "?": 6}
_MISSING_SPACE_RE = re.compile(r"([,.!?:;…])(?=[^\s])")
# Hungarian attaches suffixes to spelled-out symbols with a hyphen: the space
# that "%" → " százalék " introduces must not survive in "100%-ban".
_SUFFIX_HYPHEN_RE = re.compile(r"\s+-(?=\w)")
_LEADING_PUNCT_RE = re.compile(r"^[\s,.:;–…]+")
# The vocabulary contains these symbols, but the training transcripts spell
# them out, so a bare glyph has no reliable pronunciation.
_SYMBOL_WORDS_HU = {
    "%": " százalék ",
    "&": " és ",
    "+": " plusz ",
    "=": " egyenlő ",
    "@": " kukac ",
    "*": " csillag ",
    "§": " paragrafus ",
    "°": " fok ",
}
_HUNGARIAN = {"hu", "hun", "hungarian", "magyar"}
# Characters the training set never saw, folded onto ones it did. Parentheses
# become commas so the prosodic break survives; the rest are typographic
# variants of punctuation that is already in the vocabulary.
_VOCAB_FOLD = {
    "(": ",", ")": ",", "[": ",", "]": ",", "{": ",", "}": ",",
    "—": "–", "―": "–", "−": "–", "‒": "–",
    "“": '"', "”": '"', "‟": '"', "″": '"', "«": "»",
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'", "`": "'",
    " ": " ", " ": " ", " ": " ", "\t": " ",
    # Legacy PDF extractions substitute these for the long Hungarian vowels.
    "õ": "ő", "Õ": "ő", "û": "ű", "Û": "ű",
}
_FOLD_TABLE = str.maketrans(_VOCAB_FOLD)


def _setting(key: str, default: Any) -> Any:
    try:
        from core.settings import get

        return get(key, default)
    except Exception:
        return default


@contextlib.contextmanager
def _quiet():
    """Swallow F5-TTS' unconditional progress prints, keeping them at DEBUG."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            yield
    finally:
        captured = buffer.getvalue().strip()
        if captured:
            log.debug("f5-tts: %s", captured)


def _trim_onset_artifact(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop the leading warmup burst, keeping the first real phoneme intact."""
    max_samples = int(sample_rate * ONSET_TRIM_MAX_MS / 1000)
    window = int(sample_rate * ONSET_TRIM_WINDOW_MS / 1000)
    if len(audio) < max_samples or window <= 0:
        return audio

    region = audio[:max_samples]
    energies = [
        float(np.sqrt(np.mean(np.square(region[i : i + window]))))
        for i in range(0, len(region) - window, window)
    ]
    if not energies:
        return audio

    threshold = max(energies) * ONSET_TRIM_THRESHOLD_RATIO
    for index, energy in enumerate(energies):
        if energy > threshold:
            return audio[max(0, (index - 1) * window) :]
    return audio


class F5TTSEngine:
    engine_name = "f5"
    # F5 clones only: the mandatory reference clip already fixes the voice, so
    # there is no designed voice to pin and narrator_voice_lock is never read.
    uses_voice_lock = False
    supports_variants = True

    def __init__(self, model_path: str = "", worker_label: str = "primary"):
        self.model_path = model_path
        self.worker_label = worker_label
        self.model = None
        self.tokenizer = None
        self._vocoder = None
        self._vocab_chars: frozenset[str] = frozenset()
        self._device = "cpu"
        self._lock = threading.Lock()
        self._generating = threading.Event()
        self._cancel_generate = threading.Event()
        self._loading = False
        self._ready = False
        self._cancel_load = threading.Event()
        self._error: str | None = None
        self._resolved_model = ""
        self._sample_rate = F5_SAMPLE_RATE
        self._warned_instructs: set[str] = set()
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    # ── model files ──────────────────────────────────────────────────────────

    def _source(self) -> tuple[str, str, bool]:
        """Return (checkpoint, vocabulary, is_local) without touching the disk."""
        checkpoint = str(_setting("f5_model_file", DEFAULT_MODEL_FILE) or DEFAULT_MODEL_FILE)
        vocabulary = str(_setting("f5_vocab_file", DEFAULT_VOCAB_FILE) or DEFAULT_VOCAB_FILE)
        if str(_setting("f5_model_source", "download")).lower() == "local":
            directory = self.model_path or str(_setting("f5_model_path", "") or "")
            return os.path.join(directory, checkpoint), os.path.join(directory, vocabulary), True
        repo = str(_setting("f5_model_repo", DEFAULT_MODEL_REPO) or DEFAULT_MODEL_REPO)
        return f"{repo}/{checkpoint}", f"{repo}/{vocabulary}", False

    def _resolve_files(self) -> tuple[str, str]:
        """Materialize the checkpoint and vocabulary, downloading when needed."""
        checkpoint, vocabulary, local_only = self._source()
        if local_only:
            for path in (checkpoint, vocabulary):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"F5-TTS file not found: {path}. Check its path in Settings."
                    )
            return checkpoint, vocabulary

        from huggingface_hub import hf_hub_download

        repo = str(_setting("f5_model_repo", DEFAULT_MODEL_REPO) or DEFAULT_MODEL_REPO)
        return (
            hf_hub_download(repo, str(_setting("f5_model_file", DEFAULT_MODEL_FILE))),
            hf_hub_download(repo, str(_setting("f5_vocab_file", DEFAULT_VOCAB_FILE))),
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        checkpoint, _, local_only = self._source()
        base = {"engine": self.engine_name, "model": self._resolved_model or checkpoint}
        if self._error:
            return {**base, "state": "error", "message": self._error}
        if self._ready:
            return {
                **base,
                "state": "ready",
                "generating": self._generating.is_set(),
                "accel": {
                    "effective": "f5-tts",
                    "message": f"F5-TTS CFM/DiT on {self._device} (vocos vocoder)",
                },
            }
        if self._loading:
            return {**base, "state": "loading"}
        return {
            **base,
            "state": "not_loaded",
            "model_path": checkpoint,
            "model_exists": (os.path.isfile(checkpoint) if local_only else True),
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
            raise RuntimeError(self._error or "F5-TTS model failed to load")

    def _load(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._loading = True
            self._error = None
        try:
            import torch
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model.backbones.dit import DiT

            checkpoint, vocabulary = self._resolve_files()
            if self._cancel_load.is_set():
                log.info("F5-TTS load cancelled for import-time LLM analysis.")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with _quiet():
                vocoder = load_vocoder(vocoder_name="vocos", device=device)
                model = load_model(
                    DiT,
                    F5_V1_BASE_ARCH,
                    checkpoint,
                    mel_spec_type="vocos",
                    vocab_file=vocabulary,
                    use_ema=True,
                    device=device,
                )
            if self._cancel_load.is_set():
                del model, vocoder
                self.unload()
                return

            with open(vocabulary, encoding="utf-8") as handle:
                # A trailing newline yields an empty entry; index 0 is a space
                # and doubles as F5-TTS' unknown-character slot.
                self._vocab_chars = frozenset(
                    line for line in handle.read().split("\n") if len(line) == 1
                )
            self._device = device
            self._vocoder = vocoder
            self.model = model
            self._resolved_model = checkpoint
            self._sample_rate = F5_SAMPLE_RATE
            self._ready = True
            log.info(
                "F5-TTS ready (%s, %s, %d-character vocabulary).",
                os.path.basename(checkpoint),
                device,
                len(self._vocab_chars),
            )
        except Exception as exc:
            self._error = str(exc)
            self.model = None
            self._vocoder = None
            log.error("Failed to load F5-TTS: %s", exc)
        finally:
            self._loading = False

    def unload(self) -> None:
        self._cancel_load.set()
        self.model = None
        self.tokenizer = None
        self._vocoder = None
        self._ready = False
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
        # A second export worker would need its own copy of the weights; the
        # single resident model is shared instead and calls are serialized.
        return

    def cancel(self) -> bool:
        """Stop the run between utterances.

        F5-TTS solves an ODE inside one ``torch.inference_mode()`` call with no
        step callback, so the utterance already in flight always finishes.
        """
        if not self._generating.is_set():
            return False
        self._cancel_generate.set()
        return True

    def invalidate_voice_prompt(self, ref_audio=None, ref_text=None) -> None:
        # Each call conditions on the reference waveform directly; F5-TTS keeps
        # its own preprocessing cache keyed by the file's content hash.
        return

    def _get_voice_clone_prompt(self, ref_audio, ref_text):
        return None

    # ── text and voice preparation ───────────────────────────────────────────

    def _fold_to_vocabulary(self, text: str) -> str:
        """Lowercase and fold text so no character silently becomes a space."""
        folded = _ELLIPSIS_RE.sub("…", text).translate(_FOLD_TABLE).lower()
        if self._vocab_chars:
            dropped = {c for c in folded if c not in self._vocab_chars and not c.isspace()}
            if dropped:
                log.debug("F5-TTS: dropping out-of-vocabulary characters %s", sorted(dropped))
                folded = "".join(" " if c in dropped else c for c in folded)
        folded = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", folded)
        folded = _PUNCT_RUN_RE.sub(
            lambda run: max(
                (c for c in run.group(0) if c in _PUNCT_RANK), key=_PUNCT_RANK.get
            ),
            folded,
        )
        folded = _MISSING_SPACE_RE.sub(r"\1 ", folded)
        folded = _LEADING_PUNCT_RE.sub("", folded)
        return _WHITESPACE_RE.sub(" ", folded).strip()

    def _prepare_text(
        self,
        text: str,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None,
    ) -> str:
        # Auris enrichment tags are OmniVoice control tokens. F5-TTS has no tag
        # vocabulary, and '[' is not even in the character set, so they are
        # removed rather than translated.
        text = _BRACKET_TAG_RE.sub("", text)
        text = apply_pronunciation(text, lexicon)
        if normalize_text:
            # num2words spells Hungarian numerals correctly; the checkpoint's
            # own preprocessing example spells digits one by one instead.
            text = apply_text_normalization(text, language)
        if str(language or "").strip().lower() in _HUNGARIAN:
            for symbol, word in _SYMBOL_WORDS_HU.items():
                text = text.replace(symbol, word)
            text = _SUFFIX_HYPHEN_RE.sub("-", text)
        return self._fold_to_vocabulary(text)

    def _resolve_voice(
        self, instruct: str | None, ref_audio: str | None, ref_text: str | None
    ) -> tuple[str, str]:
        if ref_audio:
            if not str(ref_text or "").strip():
                raise RuntimeError(
                    "F5-TTS needs the exact transcript of the reference audio. "
                    "Add it next to the uploaded WAV in Voice Studio — a "
                    "mismatched transcript produces garbled speech."
                )
            return ref_audio, str(ref_text)

        default_audio = str(_setting("f5_ref_audio", "") or "")
        default_text = str(_setting("f5_ref_text", "") or "")
        if not default_audio or not default_text.strip():
            raise RuntimeError(
                "F5-TTS cannot synthesize a voice from a description. Set a "
                "narrator reference WAV and its exact transcript in "
                "Settings → F5-TTS before playback."
            )
        if not os.path.isfile(default_audio):
            raise FileNotFoundError(f"F5-TTS reference audio not found: {default_audio}")

        key = str(instruct or "")
        if key and key not in self._warned_instructs:
            self._warned_instructs.add(key)
            log.warning(
                "F5-TTS ignores voice descriptions; '%s' uses the narrator "
                "reference. Upload a reference WAV to give it its own voice.",
                key[:60],
            )
        return default_audio, default_text

    # ── cache ────────────────────────────────────────────────────────────────

    @staticmethod
    def _generation_settings() -> dict:
        return {
            "nfe_step": int(_setting("f5_nfe_step", 32)),
            "cfg_strength": float(_setting("f5_cfg_strength", 2.0)),
            "sway_sampling_coef": float(_setting("f5_sway_sampling_coef", -1.0)),
            "cross_fade_duration": float(_setting("f5_cross_fade_sec", 0.15)),
            "target_rms": float(_setting("f5_target_rms", 0.1)),
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
        variant: int = 0,
    ) -> str:
        checkpoint = (
            _setting("f5_model_repo", DEFAULT_MODEL_REPO),
            _setting("f5_model_file", DEFAULT_MODEL_FILE),
            _setting("f5_model_path", ""),
        )
        payload = (
            f"f5-v{F5_CACHE_VERSION}|{text}|{instruct}|{ref_audio}|{ref_text}|{speed:.3f}|"
            f"{language or ''}|nt={int(bool(normalize_text))}|{checkpoint}|"
            f"{cls._generation_settings()}|seed={_setting('f5_seed', -1)}|"
            f"trim={int(bool(_setting('f5_trim_onset', True)))}|"
            f"lex={lexicon_version(lexicon)}{variant_cache_tag(variant)}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def cache_path(key: str) -> str:
        return os.path.join(AUDIO_CACHE_DIR, f"{key}.wav")

    # ── synthesis ────────────────────────────────────────────────────────────

    def _load_reference(self, path: str):
        import torch

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        seconds = len(audio) / float(sample_rate or 1)
        if seconds < REFERENCE_MIN_SECONDS:
            log.warning(
                "F5-TTS reference %s is %.1fs; 5-15s clones far more reliably.",
                os.path.basename(path),
                seconds,
            )
        return torch.from_numpy(audio.T.copy()), int(sample_rate)

    def _synthesize(
        self,
        text: str,
        instruct: str | None,
        ref_audio: str,
        ref_text: str,
        speed: float,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if not self._ready or self.model is None:
            raise RuntimeError("F5-TTS is not loaded. " + (self._error or "Load it first."))

        import torch
        from f5_tts.infer.utils_infer import (
            chunk_text,
            infer_batch_process,
            preprocess_ref_audio_text,
        )

        spoken = self._prepare_text(text, language, normalize_text, lexicon)
        if not spoken:
            return np.zeros(0, dtype=np.float32)

        settings = self._generation_settings()
        if seed is None:
            seed = int(_setting("f5_seed", -1))

        self._generating.set()
        try:
            with self._lock:
                with _quiet():
                    reference_path, reference_text = preprocess_ref_audio_text(
                        ref_audio, self._fold_to_vocabulary(ref_text), show_info=lambda *_: None
                    )
                    reference = self._load_reference(reference_path)
                    # F5-TTS sizes each chunk against the reference so the
                    # conditioned segment stays inside the model's 22s window.
                    audio_seconds = reference[0].shape[-1] / max(reference[1], 1)
                    max_chars = int(
                        len(reference_text.encode("utf-8"))
                        / max(audio_seconds, 1e-6)
                        * max(22 - audio_seconds, 1.0)
                        * speed
                    )
                    batches = chunk_text(spoken, max_chars=max(max_chars, 40))
                    if not batches:
                        return np.zeros(0, dtype=np.float32)
                    if seed >= 0:
                        torch.manual_seed(seed)
                    wave, sample_rate, _ = next(
                        infer_batch_process(
                            reference,
                            reference_text,
                            batches,
                            self.model,
                            self._vocoder,
                            mel_spec_type="vocos",
                            progress=None,
                            speed=speed,
                            device=self._device,
                            **settings,
                        )
                    )
        finally:
            self._generating.clear()

        if wave is None:
            return np.zeros(0, dtype=np.float32)
        audio = np.asarray(wave, dtype=np.float32)
        if bool(_setting("f5_trim_onset", True)):
            audio = _trim_onset_artifact(audio, int(sample_rate))
        return audio

    # ── public API ───────────────────────────────────────────────────────────

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
        speaker: str | None = None,
        variant: int = 0,
    ) -> dict:
        if normalize_text is None:
            normalize_text = bool(_setting("normalize_text", True))
        ref_audio, ref_text = self._resolve_voice(instruct, ref_audio, ref_text)
        key = self.cache_key(
            text,
            instruct,
            ref_audio,
            speed,
            ref_text=ref_text,
            language=language,
            normalize_text=bool(normalize_text),
            lexicon=lexicon,
            variant=variant,
        )
        path = self.cache_path(key)
        if os.path.exists(path):
            data, sample_rate = sf.read(path)
            return {
                "audio_path": path,
                "duration_sec": len(data) / sample_rate,
                "cache_hit": True,
                "cache_key": key,
            }
        # A configured fixed seed would return the same take every time, so
        # alternatives derive their own. Variant 0 keeps the configured seed.
        seed = variant_seed(int(_setting("f5_seed", -1)), variant) if variant else None
        audio = self._synthesize(
            text, instruct, ref_audio, ref_text, speed, language, bool(normalize_text),
            lexicon=lexicon,
            seed=seed,
        )
        _write_audio_atomic(path, audio, self._sample_rate)
        return {
            "audio_path": path,
            "duration_sec": len(audio) / self._sample_rate,
            "cache_hit": False,
            "cache_key": key,
        }

    def generate_many(
        self,
        items: list[dict],
        num_step: int | None = None,
        batch_size: int | None = None,
        on_item=None,
        on_status=None,
    ) -> list[dict]:
        results: list[dict] = []
        total = len(items)
        self._cancel_generate.clear()
        for index, item in enumerate(items):
            if self._cancel_generate.is_set():
                log.info("F5-TTS run cancelled after %d/%d utterances.", index, total)
                break
            if on_status is not None:
                on_status(f"F5-TTS utterance {index + 1}/{total}…")
            result = self.generate(
                text=item["text"],
                instruct=item.get("instruct"),
                ref_audio=item.get("ref_audio"),
                ref_text=item.get("ref_text"),
                speed=float(item.get("speed") or 1.0),
                language=item.get("language"),
                normalize_text=item.get("normalize_text"),
                lexicon=item.get("lexicon"),
            )
            results.append(result)
            if on_item is not None:
                on_item(index, result)
        self._cancel_generate.clear()
        return results

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
        # casting; F5 takes its voice entirely from the reference clip.
        return self.generate(
            sample_text,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            normalize_text=normalize_text,
        )
