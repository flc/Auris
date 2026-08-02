"""Piper TTS engine — fast CPU synthesis with distinct per-character voices.

Piper is a different trade from the other local engines. It cannot clone and it
has no expression control, but it is ONNX on the CPU: no VRAM, no torch, and
roughly 30x faster than real time. That makes it the practical choice for
drafting a whole book, for machines without a usable GPU, and for previewing
before committing to a slow engine.

What it *can* do that F5 cannot is speak in more than one voice for free. Each
Piper voice is its own small model, so Auris casts characters across the
configured voices deterministically — matched by gender where the character
description states one — instead of reading every character in the narrator's
timbre.

Two details drive the implementation:

* Piper models are 22.05 kHz while the rest of Auris (notably ``exporter``,
  which discards the sample rate it reads and writes chapters at a fixed rate)
  assumes 24 kHz. Output is resampled here so nothing downstream has to care.
* espeak-ng phonemizes anything it is given, so an Auris enrichment tag left in
  the text is pronounced out loud — ``[laughter]`` becomes "lˈɑuɡhtɛr".
"""

from __future__ import annotations

import gc
import hashlib
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
    SAMPLE_RATE,
    _write_audio_atomic,
    apply_text_normalization,
    variant_cache_tag,
)

log = logging.getLogger(__name__)

VOICE_REPO = "rhasspy/piper-voices"
PIPER_CACHE_VERSION = 1
DEFAULT_NARRATOR_VOICE = "hu_HU-anna-medium"
DEFAULT_CHARACTER_VOICES = "hu_HU-berta-medium,hu_HU-imre-medium"
# Keeping more than a handful resident buys nothing: a medium voice is ~60 MB
# and a book rarely casts beyond the configured list.
MAX_RESIDENT_VOICES = 8

# Gender is not in the upstream voices.json, so it is recorded here for casting.
# The values were checked against each voice's median F0 (measured on the same
# Hungarian sentence): anna 173 Hz, berta 170 Hz, imre 112 Hz.
VOICE_GENDER = {
    "hu_HU-anna-medium": "female",
    "hu_HU-berta-medium": "female",
    "hu_HU-imre-medium": "male",
}

_VOICE_KEY_RE = re.compile(r"^(?P<locale>[a-z]{2,3}_[A-Z]{2})-(?P<name>.+)-(?P<quality>[a-z_]+)$")
_BRACKET_TAG_RE = re.compile(r"\[([a-z0-9_-]+)\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_FEMALE_RE = re.compile(r"\b(female|woman|girl|n[őo]i?)\b", re.IGNORECASE)
_MALE_RE = re.compile(r"\b(male|man|boy|f[ée]rfi)\b", re.IGNORECASE)
_HUNGARIAN = {"hu", "hun", "hungarian", "magyar"}
# Legacy PDF extractions substitute these for the long Hungarian vowels; espeak
# would otherwise voice them as Portuguese/French letters.
_HU_GLYPH_FIX = str.maketrans({"õ": "ő", "Õ": "Ő", "û": "ű", "Û": "Ű"})


def _setting(key: str, default: Any) -> Any:
    try:
        from core.settings import get

        return get(key, default)
    except Exception:
        return default


def voice_repo_paths(key: str) -> tuple[str, str]:
    """Map ``hu_HU-anna-medium`` to its model and config paths in the voice repo.

    The layout is ``<lang>/<locale>/<name>/<quality>/<key>.onnx``, so any voice
    in the upstream repository can be named without a hard-coded table.
    """
    match = _VOICE_KEY_RE.match(key.strip())
    if not match:
        raise ValueError(
            f"Not a Piper voice name: {key!r}. Expected e.g. 'hu_HU-anna-medium'."
        )
    locale = match.group("locale")
    directory = f"{locale.split('_')[0]}/{locale}/{match.group('name')}/{match.group('quality')}"
    return f"{directory}/{key}.onnx", f"{directory}/{key}.onnx.json"


def is_voice_name(key: str) -> bool:
    """Whether ``key`` is a well-formed Piper voice name."""
    return bool(_VOICE_KEY_RE.match(str(key or "").strip()))


def _voice_list(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _instruct_gender(instruct: str | None) -> str | None:
    text = str(instruct or "")
    if _FEMALE_RE.search(text):
        return "female"
    if _MALE_RE.search(text):
        return "male"
    return None


class PiperTTSEngine:
    engine_name = "piper"
    # The voice is the model file. There is no designed voice to pin, so the
    # narrator voice lock never applies.
    uses_voice_lock = False
    # VITS samples its duration and acoustic noise per call, so repeat calls
    # already differ; only the cache key has to separate the takes.
    supports_variants = True

    def __init__(self, model_path: str = "", worker_label: str = "primary"):
        self.model_path = model_path
        self.worker_label = worker_label
        self.model = None
        self.tokenizer = None
        self._voices: dict[str, Any] = {}
        self._voice_rates: dict[str, int] = {}
        self._lock = threading.Lock()
        self._generating = threading.Event()
        self._cancel_generate = threading.Event()
        self._loading = False
        self._ready = False
        self._cancel_load = threading.Event()
        self._error: str | None = None
        self._resolved_model = ""
        self._sample_rate = SAMPLE_RATE
        self._warned_cloning = False
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    # ── voice configuration ──────────────────────────────────────────────────

    @staticmethod
    def _narrator_voice() -> str:
        return str(
            _setting("piper_narrator_voice", DEFAULT_NARRATOR_VOICE) or DEFAULT_NARRATOR_VOICE
        )

    @classmethod
    def _character_voices(cls) -> list[str]:
        """Voices available for dialogue, never including the narrator's.

        Sharing the narrator's timbre with a character is the one casting
        mistake that actively hurts an audiobook, so it is excluded whenever
        anything else is configured.
        """
        configured = _voice_list(
            _setting("piper_character_voices", DEFAULT_CHARACTER_VOICES)
        )
        narrator = cls._narrator_voice()
        distinct = [voice for voice in configured if voice != narrator]
        return distinct or configured

    @classmethod
    def _configured_voices(cls) -> list[str]:
        seen = dict.fromkeys([cls._narrator_voice(), *cls._character_voices()])
        return [voice for voice in seen if voice]

    @classmethod
    def voice_for(cls, instruct: str | None, speaker: str | None) -> str:
        """Cast one segment, deterministically and stably across runs."""
        narrator = cls._narrator_voice()
        if not speaker:
            return narrator

        candidates = cls._character_voices()
        if not candidates:
            return narrator

        if bool(_setting("piper_match_gender", True)):
            gender = _instruct_gender(instruct)
            if gender:
                matching = [v for v in candidates if VOICE_GENDER.get(v) == gender]
                if matching:
                    candidates = matching

        # Keyed on the character name, not the description: editing a
        # character's voice text in Voice Studio should not recast them.
        digest = hashlib.md5(str(speaker).strip().lower().encode("utf-8")).hexdigest()
        return candidates[int(digest[:8], 16) % len(candidates)]

    # ── model files ──────────────────────────────────────────────────────────

    def _resolve_voice_files(self, key: str) -> tuple[str, str]:
        model, config = voice_repo_paths(key)
        if str(_setting("piper_voice_source", "download")).lower() == "local":
            directory = self.model_path or str(_setting("piper_voice_dir", "") or "")
            local_model = os.path.join(directory, f"{key}.onnx")
            local_config = os.path.join(directory, f"{key}.onnx.json")
            for path in (local_model, local_config):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Piper voice file not found: {path}. Check the voice "
                        "directory in Settings."
                    )
            return local_model, local_config

        from huggingface_hub import hf_hub_download

        repo = str(_setting("piper_voice_repo", VOICE_REPO) or VOICE_REPO)
        return hf_hub_download(repo, model), hf_hub_download(repo, config)

    def _get_voice(self, key: str):
        """Return a loaded voice, fetching and caching it on first use."""
        voice = self._voices.get(key)
        if voice is not None:
            return voice

        from piper import PiperVoice

        model, config = self._resolve_voice_files(key)
        voice = PiperVoice.load(model, config_path=config)
        if len(self._voices) >= MAX_RESIDENT_VOICES:
            self._voices.pop(next(iter(self._voices)), None)
        self._voices[key] = voice
        self._voice_rates[key] = int(voice.config.sample_rate)
        log.info(
            "Piper voice ready: %s (%d Hz, %d speaker(s))",
            key,
            voice.config.sample_rate,
            voice.config.num_speakers,
        )
        return voice

    # ── lifecycle ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        narrator = self._narrator_voice()
        base = {"engine": self.engine_name, "model": self._resolved_model or narrator}
        if self._error:
            return {**base, "state": "error", "message": self._error}
        if self._ready:
            return {
                **base,
                "state": "ready",
                "generating": self._generating.is_set(),
                "accel": {
                    "effective": "onnxruntime",
                    "message": (
                        f"Piper on CPU — {len(self._voices)} voice(s) loaded, "
                        f"cast across {len(self._configured_voices())}"
                    ),
                },
            }
        if self._loading:
            return {**base, "state": "loading"}
        return {**base, "state": "not_loaded", "model_path": narrator, "model_exists": True}

    def load_async(self) -> None:
        if self._ready or self._loading:
            return
        self._cancel_load.clear()
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self) -> None:
        self._cancel_load.clear()
        self._load()
        if not self._ready:
            raise RuntimeError(self._error or "Piper voices failed to load")

    def _load(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._loading = True
            self._error = None
        try:
            narrator = self._narrator_voice()
            self._get_voice(narrator)
            if self._cancel_load.is_set():
                log.info("Piper load cancelled for import-time LLM analysis.")
                self.unload()
                return
            self._resolved_model = narrator
            self._ready = True
            log.info(
                "Piper ready (narrator %s, character voices: %s).",
                narrator,
                ", ".join(self._character_voices()) or "none",
            )
        except Exception as exc:
            self._error = str(exc)
            self._voices.clear()
            log.error("Failed to load Piper: %s", exc)
        finally:
            self._loading = False

    def unload(self) -> None:
        self._cancel_load.set()
        self._voices.clear()
        self._voice_rates.clear()
        self.model = None
        self._ready = False
        gc.collect()

    def reload(self) -> None:
        self.unload()
        self._error = None
        self.load_async()

    def set_dedicated_cuda_stream(self, enabled: bool) -> None:
        # Piper runs on the CPU; there is no stream to dedicate.
        return

    def cancel(self) -> bool:
        """Stop between utterances.

        One Piper utterance is a single ONNX call taking tens of milliseconds,
        so there is nothing worth interrupting inside it.
        """
        if not self._generating.is_set():
            return False
        self._cancel_generate.set()
        return True

    def invalidate_voice_prompt(self, ref_audio=None, ref_text=None) -> None:
        # Piper has no reference conditioning to invalidate.
        return

    def _get_voice_clone_prompt(self, ref_audio, ref_text):
        return None

    # ── text preparation ─────────────────────────────────────────────────────

    def _prepare_text(
        self,
        text: str,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None,
    ) -> str:
        # espeak-ng phonemizes whatever it receives, so a leftover enrichment
        # tag is spoken as a word rather than ignored.
        text = _BRACKET_TAG_RE.sub("", text)
        if str(language or "").strip().lower() in _HUNGARIAN:
            text = text.translate(_HU_GLYPH_FIX)
        text = apply_pronunciation(text, lexicon)
        if normalize_text:
            # espeak has its own number rules, but going through Auris' shared
            # normalizer keeps a chapter identical across engines.
            text = apply_text_normalization(text, language)
        return _WHITESPACE_RE.sub(" ", text).strip()

    # ── cache ────────────────────────────────────────────────────────────────

    @staticmethod
    def _synthesis_settings() -> dict:
        def optional(key: str) -> float | None:
            value = _setting(key, -1.0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None

        return {
            "noise_scale": optional("piper_noise_scale"),
            "noise_w_scale": optional("piper_noise_w_scale"),
            "length_scale_base": float(_setting("piper_length_scale", 1.0)),
            "normalize_audio": bool(_setting("piper_normalize_audio", True)),
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
        speaker: str | None = None,
        variant: int = 0,
    ) -> str:
        # The voice, not the instruct, is what actually changes the audio — but
        # the instruct feeds gender-matched casting, so both belong in the key.
        voice = cls.voice_for(instruct, speaker)
        payload = (
            f"piper-v{PIPER_CACHE_VERSION}|{text}|{voice}|{speed:.3f}|{language or ''}|"
            f"nt={int(bool(normalize_text))}|{cls._synthesis_settings()}|"
            f"lex={lexicon_version(lexicon)}{variant_cache_tag(variant)}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def cache_path(key: str) -> str:
        return os.path.join(AUDIO_CACHE_DIR, f"{key}.wav")

    # ── synthesis ────────────────────────────────────────────────────────────

    def _synthesize(
        self,
        text: str,
        voice_key: str,
        speed: float,
        language: str | None,
        normalize_text: bool,
        lexicon: str | None = None,
    ) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Piper is not loaded. " + (self._error or "Load it first."))

        from piper import SynthesisConfig

        spoken = self._prepare_text(text, language, normalize_text, lexicon)
        if not spoken:
            return np.zeros(0, dtype=np.float32)

        settings = self._synthesis_settings()
        # length_scale is a duration multiplier, so it is the inverse of speed.
        length_scale = settings["length_scale_base"] / max(speed, 0.05)
        config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=settings["noise_scale"],
            noise_w_scale=settings["noise_w_scale"],
            normalize_audio=settings["normalize_audio"],
        )

        self._generating.set()
        try:
            with self._lock:
                voice = self._get_voice(voice_key)
                chunks = list(voice.synthesize(spoken, config))
        finally:
            self._generating.clear()

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return self._to_pipeline_rate(audio, int(chunks[0].sample_rate))

    @staticmethod
    def _to_pipeline_rate(audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample to Auris' 24 kHz.

        Piper voices are 22.05 kHz, but ``exporter`` drops the rate it reads
        from each cached WAV and writes the merged chapter at a fixed 24 kHz.
        Left alone, a Piper chapter would export ~9% fast and a semitone sharp.
        """
        if source_rate == SAMPLE_RATE or len(audio) == 0:
            return np.asarray(audio, dtype=np.float32)

        import librosa

        resampled = librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=source_rate,
            target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        )
        return np.asarray(resampled, dtype=np.float32)

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
        if ref_audio and not self._warned_cloning:
            self._warned_cloning = True
            log.warning(
                "Piper cannot clone; uploaded reference clips are ignored and "
                "voices are cast from the configured Piper voice list instead."
            )
        voice_key = self.voice_for(instruct, speaker)
        key = self.cache_key(
            text,
            instruct,
            ref_audio,
            speed,
            ref_text=ref_text,
            language=language,
            normalize_text=bool(normalize_text),
            lexicon=lexicon,
            speaker=speaker,
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
        audio = self._synthesize(
            text, voice_key, speed, language, bool(normalize_text), lexicon=lexicon
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
                log.info("Piper run cancelled after %d/%d utterances.", index, total)
                break
            if on_status is not None:
                on_status(f"Piper utterance {index + 1}/{total}…")
            result = self.generate(
                text=item["text"],
                instruct=item.get("instruct"),
                ref_audio=item.get("ref_audio"),
                ref_text=item.get("ref_text"),
                speed=float(item.get("speed") or 1.0),
                language=item.get("language"),
                normalize_text=item.get("normalize_text"),
                lexicon=item.get("lexicon"),
                speaker=item.get("speaker"),
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
        # Voice Studio must preview the voice playback will actually use, so a
        # character preview has to be cast the same way — hence `speaker`.
        return self.generate(
            sample_text,
            instruct=instruct,
            language=language,
            normalize_text=normalize_text,
            speaker=speaker,
        )
