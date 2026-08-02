"""ElevenLabs cloud TTS engine.

Minimal integration: the whole book is read by one configured voice.  Auris
describes a speaker with an ``instruct`` string and clones from a reference
WAV, while ElevenLabs addresses a speaker by ``voice_id``.  Bridging those two
models (voice design from the instruct text, instant voice cloning from the
reference clip) needs a persistent instruct/clip -> voice_id map and is kept
out of this first version.  ``_voice_id_for()`` is the single seam where that
mapping will land; everything else already routes through it.

Unlike the local engines there is no model to load, so the lifecycle calls are
either no-ops or one cheap HTTP round trip that validates the API key.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import soundfile as sf

from core.pronunciation import apply_pronunciation, lexicon_version
from core.tts_engine import (
    AUDIO_CACHE_DIR,
    SAMPLE_RATE,
    _write_audio_atomic,
    apply_text_normalization,
    variant_cache_tag,
    variant_seed,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_CACHE_VERSION = 1

# Formats the engine can decode back into a cache WAV. ulaw/opus containers are
# deliberately excluded rather than silently mishandled.
SUPPORTED_OUTPUT_FORMATS = (
    "mp3_22050_32",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
)

# The per-request speed control and language_code are model dependent. When the
# API rejects a request because of them, the call is retried without them.
_OPTIONAL_PAYLOAD_FIELDS = ("language_code", "voice_settings.speed", "seed")
_SPEED_RANGE = (0.7, 1.2)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SEC = 1.5

# Auris enrichment emits OmniVoice non-verbal tags. ElevenLabs v2 models would
# read them out loud, so they are stripped. (v3 audio tags are a phase-2 map.)
_BRACKET_TAG_RE = re.compile(r"\[([a-z0-9_-]+)\]", re.IGNORECASE)


class ElevenLabsAPIError(RuntimeError):
    """HTTP-level failure carrying the status code for retry decisions."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _setting(key: str, default: Any) -> Any:
    try:
        from core.settings import get

        return get(key, default)
    except Exception:
        return default


def _api_key() -> str:
    """Environment wins over settings.json so keys can stay out of the file."""
    return str(
        os.environ.get("ELEVENLABS_API_KEY")
        or _setting("elevenlabs_api_key", "")
        or ""
    ).strip()


def _base_url() -> str:
    return str(_setting("elevenlabs_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")


def _output_format() -> str:
    value = str(_setting("elevenlabs_output_format", DEFAULT_OUTPUT_FORMAT) or "").strip()
    return value if value in SUPPORTED_OUTPUT_FORMATS else DEFAULT_OUTPUT_FORMAT


def _strip_inline_tags(text: str) -> str:
    return _BRACKET_TAG_RE.sub("", text)


def _sends_language_code(model_id: str) -> bool:
    """Only the turbo/flash v2.5 models accept an explicit language_code."""
    value = str(model_id or "").lower()
    return "turbo" in value or "flash" in value


def _decode_audio(data: bytes, output_format: str) -> tuple[np.ndarray, int]:
    """Return mono float32 samples and their sample rate."""
    if output_format.startswith("pcm_"):
        rate = int(output_format.split("_")[1])
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, rate
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        # libsndfile builds without MP3 support fall back to pydub + ffmpeg,
        # which the project already requires for MP3 export.
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(data))
        rate = segment.frame_rate
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).mean(axis=1)
        audio = samples / float(1 << (8 * segment.sample_width - 1))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(rate)


def _drop_optional_field(payload: dict, field: str) -> bool:
    if "." in field:
        parent, child = field.split(".", 1)
        section = payload.get(parent)
        if isinstance(section, dict) and child in section:
            section.pop(child)
            return True
        return False
    if field in payload:
        payload.pop(field)
        return True
    return False


class ElevenLabsTTSEngine:
    engine_name = "elevenlabs"
    # A voice id is already a stable identity on the provider's side, so there
    # is nothing to pin and narrator_voice_lock is never read.
    uses_voice_lock = False
    # The API takes a seed, so alternative takes are reproducible — and, being
    # a metered engine, each one is billed like any other segment.
    supports_variants = True

    def __init__(self, model_path: str = "", worker_label: str = "primary"):
        # model_path/worker_label exist only so the engine stays interchangeable
        # with the local ones; a cloud engine has no local weights.
        self.model_path = model_path
        self.worker_label = worker_label
        self.model = None
        self.tokenizer = None
        self._lock = threading.RLock()
        self._loading = False
        self._ready = False
        self._error: str | None = None
        self._generating = threading.Event()
        self._cancel = threading.Event()
        self._inflight: set = set()
        self._inflight_lock = threading.Lock()
        self._quota: dict = {}
        self._characters_sent = 0
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        voice_id = str(_setting("elevenlabs_voice_id", "") or "").strip()
        model_id = str(_setting("elevenlabs_model_id", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID)
        base = {
            "engine": self.engine_name,
            "model": model_id,
            "voice_id": voice_id,
            "characters_sent": self._characters_sent,
        }
        if self._quota:
            base["quota"] = dict(self._quota)
        if not _api_key():
            return {
                **base,
                "state": "error",
                "message": "ElevenLabs API key is not set. Add it in Settings → Speech engines.",
            }
        if not voice_id:
            return {
                **base,
                "state": "error",
                "message": "No ElevenLabs voice selected. Paste a voice ID in Settings → Speech engines.",
            }
        if self._error:
            return {**base, "state": "error", "message": self._error}
        if self._ready:
            return {
                **base,
                "state": "ready",
                "generating": self._generating.is_set(),
                "accel": {"effective": "cloud", "message": "ElevenLabs HTTP API"},
            }
        if self._loading:
            return {**base, "state": "loading"}
        return {**base, "state": "not_loaded"}

    def load_async(self) -> None:
        # A failed key check is sticky: /api/tts/status polls this call, and an
        # invalid key must not turn into one 401 per poll. Reload clears it.
        if self._ready or self._loading or self._error:
            return
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self) -> None:
        self._load()
        if not self._ready:
            raise RuntimeError(self._error or "ElevenLabs TTS is not available")

    def _load(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._loading = True
            self._error = None
        try:
            if not _api_key():
                raise RuntimeError("ElevenLabs API key is not set.")
            # Snapshot what is being validated: settings can change under a
            # background load, and the log should name what was checked.
            voice_id = str(_setting("elevenlabs_voice_id", "") or "").strip()
            model_id = str(_setting("elevenlabs_model_id", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID)
            output_format = _output_format()
            if not voice_id:
                raise RuntimeError("No ElevenLabs voice ID is configured.")
            self._probe_subscription()
            self._ready = True
            self._cancel.clear()
            log.info(
                "ElevenLabs TTS ready (model=%s, voice=%s, format=%s).",
                model_id,
                voice_id,
                output_format,
            )
        except Exception as exc:
            self._error = str(exc)
            log.error("ElevenLabs TTS unavailable: %s", exc)
        finally:
            self._loading = False

    def _probe_subscription(self) -> None:
        """Validate the key and record the character quota when readable."""
        try:
            raw = self._request("GET", "/v1/user/subscription")
        except ElevenLabsAPIError as exc:
            if exc.status == 401:
                raise RuntimeError(f"ElevenLabs rejected the API key: {exc}") from exc
            # A key scoped to text-to-speech only cannot read the subscription.
            # That is not a reason to refuse synthesis.
            log.info("ElevenLabs quota unreadable (%s); continuing without it.", exc)
            self._quota = {}
            return
        try:
            data = json.loads(raw.decode("utf-8"))
            used = int(data.get("character_count", 0))
            limit = int(data.get("character_limit", 0))
            self._quota = {
                "character_count": used,
                "character_limit": limit,
                "characters_left": max(0, limit - used),
                "tier": str(data.get("tier") or ""),
            }
        except (ValueError, TypeError, AttributeError):
            self._quota = {}

    def unload(self) -> None:
        # Nothing is resident. Dropping ready state keeps the shared lifecycle
        # honest: the next status poll re-validates the key.
        self.cancel()
        self._ready = False

    def reload(self) -> None:
        self.unload()
        self._error = None
        self._cancel.clear()
        self.load_async()

    def cancel(self) -> bool:
        if not self._generating.is_set():
            return False
        self._cancel.set()
        with self._inflight_lock:
            pending = list(self._inflight)
        for response in pending:
            try:
                response.close()
            except Exception:
                pass
        return True

    def set_dedicated_cuda_stream(self, enabled: bool) -> None:
        return

    def invalidate_voice_prompt(self, ref_audio=None, ref_text=None) -> None:
        return

    def _get_voice_clone_prompt(self, ref_audio, ref_text):
        return None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        accept: str = "application/json",
    ) -> bytes:
        url = f"{_base_url()}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("xi-api-key", _api_key())
        request.add_header("Accept", accept)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            timeout = float(_setting("elevenlabs_timeout_sec", 120) or 120)
        except (TypeError, ValueError):
            timeout = 120.0
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise ElevenLabsAPIError(
                f"HTTP {exc.code} from {path}: {detail or exc.reason}", exc.code
            ) from exc
        except urllib.error.URLError as exc:
            raise ElevenLabsAPIError(f"Cannot reach {url}: {exc.reason}") from exc
        with self._inflight_lock:
            self._inflight.add(response)
        try:
            return response.read()
        except Exception as exc:
            if self._cancel.is_set():
                raise ElevenLabsAPIError("ElevenLabs request cancelled") from exc
            raise ElevenLabsAPIError(f"Reading {path} failed: {exc}") from exc
        finally:
            with self._inflight_lock:
                self._inflight.discard(response)
            try:
                response.close()
            except Exception:
                pass

    def _post_speech(self, voice_id: str, payload: dict, output_format: str) -> bytes:
        path = f"/v1/text-to-speech/{voice_id}?output_format={output_format}"
        body = json.loads(json.dumps(payload))  # defensive deep copy
        optional_left = list(_OPTIONAL_PAYLOAD_FIELDS)
        last_error: ElevenLabsAPIError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            if self._cancel.is_set():
                raise ElevenLabsAPIError("ElevenLabs request cancelled")
            try:
                return self._request("POST", path, body=body, accept="audio/*")
            except ElevenLabsAPIError as exc:
                last_error = exc
                # Model-dependent knobs are the usual cause of a 4xx here. Shed
                # them one at a time before giving up on the request.
                if exc.status in (400, 422) and optional_left:
                    dropped = False
                    while optional_left and not dropped:
                        dropped = _drop_optional_field(body, optional_left.pop(0))
                    if dropped:
                        log.warning(
                            "ElevenLabs rejected an optional field (%s); retrying without it.",
                            exc,
                        )
                        continue
                if exc.status not in _RETRY_STATUSES:
                    raise
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                delay = _BACKOFF_BASE_SEC * (2**attempt)
                log.warning("ElevenLabs %s; retrying in %.1fs.", exc, delay)
                if self._cancel.wait(delay):
                    raise ElevenLabsAPIError("ElevenLabs request cancelled") from exc
        raise last_error or ElevenLabsAPIError("ElevenLabs request failed")

    # ── synthesis ────────────────────────────────────────────────────────────

    @staticmethod
    def _voice_settings() -> dict:
        def number(key: str, default: float, low: float, high: float) -> float:
            try:
                return max(low, min(float(_setting(key, default)), high))
            except (TypeError, ValueError):
                return default

        return {
            "stability": number("elevenlabs_stability", 0.5, 0.0, 1.0),
            "similarity_boost": number("elevenlabs_similarity_boost", 0.75, 0.0, 1.0),
            "style": number("elevenlabs_style", 0.0, 0.0, 1.0),
            "use_speaker_boost": bool(_setting("elevenlabs_speaker_boost", True)),
        }

    @staticmethod
    def _voice_id_for(instruct: str | None, ref_audio: str | None) -> str:
        """Resolve the ElevenLabs voice for a segment.

        One configured voice for now. Per-character voices, voice design from
        ``instruct`` and instant cloning from ``ref_audio`` all plug in here;
        because the resolved ID is part of the cache key, that change will
        invalidate exactly the segments whose voice actually differs.
        """
        return str(_setting("elevenlabs_voice_id", "") or "").strip()

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
        # instruct/ref_audio are intentionally absent: they do not reach the API
        # yet, and keying on them would pay for the same audio once per
        # character. _voice_id_for() carries whatever actually varies.
        voice_id = cls._voice_id_for(instruct, ref_audio)
        model_id = str(_setting("elevenlabs_model_id", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID)
        payload = (
            f"elevenlabs-v{ELEVENLABS_CACHE_VERSION}|{text}|{voice_id}|{model_id}|"
            f"{cls._voice_settings()}|{speed:.3f}|{language or ''}|"
            f"nt={int(bool(normalize_text))}|{_output_format()}|"
            f"lex={lexicon_version(lexicon)}{variant_cache_tag(variant)}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def cache_path(key: str) -> str:
        return os.path.join(AUDIO_CACHE_DIR, f"{key}.wav")

    def _spoken_text(
        self,
        text: str,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None,
    ) -> str:
        spoken = apply_pronunciation(text, lexicon)
        if normalize_text:
            spoken = apply_text_normalization(spoken, language)
        # Stripping a tag leaves the spaces that surrounded it behind.
        return re.sub(r"\s+", " ", _strip_inline_tags(spoken)).strip()

    def _synthesize(
        self,
        text: str,
        instruct: str | None,
        ref_audio: str | None,
        speed: float,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None = None,
        variant: int = 0,
    ) -> tuple[np.ndarray, int]:
        if not self._ready:
            raise RuntimeError(
                "ElevenLabs TTS is not ready. " + (self._error or "Check the API key in Settings.")
            )
        voice_id = self._voice_id_for(instruct, ref_audio)
        if not voice_id:
            raise RuntimeError("No ElevenLabs voice ID is configured.")
        spoken = self._spoken_text(text, language, normalize_text, lexicon)
        if not spoken:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE

        model_id = str(_setting("elevenlabs_model_id", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID)
        voice_settings = self._voice_settings()
        clamped = max(_SPEED_RANGE[0], min(float(speed or 1.0), _SPEED_RANGE[1]))
        if abs(clamped - 1.0) > 0.01:
            voice_settings["speed"] = round(clamped, 3)
        payload: dict = {"text": spoken, "model_id": model_id, "voice_settings": voice_settings}
        if language and _sends_language_code(model_id):
            payload["language_code"] = str(language).strip().lower()[:2]
        if variant:
            # Pinning the seed makes an alternative take reproducible, so the
            # same variant does not have to be paid for twice.
            payload["seed"] = variant_seed(0, variant)

        output_format = _output_format()
        self._generating.set()
        try:
            data = self._post_speech(voice_id, payload, output_format)
        finally:
            self._generating.clear()
        self._characters_sent += len(spoken)
        audio, rate = _decode_audio(data, output_format)
        if len(audio) == 0:
            raise RuntimeError("ElevenLabs returned an empty audio stream")
        return audio, rate

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
        key = self.cache_key(
            text,
            instruct,
            ref_audio,
            float(speed or 1.0),
            ref_text=ref_text,
            language=language,
            normalize_text=bool(normalize_text),
            lexicon=lexicon,
            variant=variant,
        )
        path = self.cache_path(key)
        if os.path.exists(path):
            data, rate = sf.read(path)
            return {
                "audio_path": path,
                "duration_sec": len(data) / rate,
                "cache_hit": True,
                "cache_key": key,
            }
        audio, rate = self._synthesize(
            text,
            instruct,
            ref_audio,
            float(speed or 1.0),
            language,
            bool(normalize_text),
            lexicon=lexicon,
            variant=variant,
        )
        _write_audio_atomic(path, audio, rate)
        return {
            "audio_path": path,
            "duration_sec": len(audio) / rate,
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
        # Serial for now: one HTTP request per segment. Concurrency is the
        # obvious next win, but it needs rate-limit budgeting per plan tier.
        results: list[dict] = []
        total = len(items)
        for index, item in enumerate(items):
            if self._cancel.is_set():
                raise RuntimeError("ElevenLabs generation cancelled")
            if on_status is not None:
                on_status(f"ElevenLabs utterance {index + 1}/{total}…")
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
        # casting; this engine reads the whole book with one configured voice.
        return self.generate(
            sample_text,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            normalize_text=normalize_text,
        )
