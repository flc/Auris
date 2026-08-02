"""Unit tests for the Piper engine.

The behaviour worth pinning is the casting: which voice a segment gets has to
be stable across runs and processes, or a re-export re-renders a book in
different voices. The rest is text handling — espeak pronounces anything it is
handed, so a leftover enrichment tag is spoken as a word.
"""

import unittest
from unittest.mock import patch

import numpy as np

from core.piper_engine import (
    DEFAULT_CHARACTER_VOICES,
    DEFAULT_NARRATOR_VOICE,
    PiperTTSEngine,
    is_voice_name,
    voice_repo_paths,
)
from core.tts_router import ENGINE_NAMES, TTSEngineRouter, engine_uses_voice_lock

DEFAULTS = {
    "piper_narrator_voice": DEFAULT_NARRATOR_VOICE,
    "piper_character_voices": DEFAULT_CHARACTER_VOICES,
    "piper_match_gender": True,
}


def _with_settings(**overrides):
    values = {**DEFAULTS, **overrides}
    return patch(
        "core.piper_engine._setting",
        side_effect=lambda key, default: values.get(key, default),
    )


class VoiceNameTests(unittest.TestCase):
    def test_repo_path_is_derived_from_the_voice_name(self):
        self.assertEqual(
            voice_repo_paths("hu_HU-anna-medium"),
            (
                "hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx",
                "hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx.json",
            ),
        )

    def test_derivation_generalizes_beyond_hungarian(self):
        model, _ = voice_repo_paths("en_US-lessac-high")
        self.assertEqual(model, "en/en_US/lessac/high/en_US-lessac-high.onnx")

    def test_malformed_names_are_rejected_rather_than_downloaded(self):
        for bad in ("", "anna", "hu-anna-medium", "hu_HU-anna", "../../etc/passwd"):
            self.assertFalse(is_voice_name(bad), bad)
            with self.assertRaises(ValueError):
                voice_repo_paths(bad)


class CastingTests(unittest.TestCase):
    def test_narration_uses_the_configured_narrator_voice(self):
        with _with_settings():
            self.assertEqual(
                PiperTTSEngine.voice_for("male, elderly", None), DEFAULT_NARRATOR_VOICE
            )

    def test_characters_never_get_the_narrator_voice(self):
        with _with_settings():
            for name in ("Éva", "Gergely", "Bornemissza", "Vicuska", "Dobó"):
                self.assertNotEqual(
                    PiperTTSEngine.voice_for("male, elderly", name),
                    DEFAULT_NARRATOR_VOICE,
                )

    def test_gender_in_the_description_picks_a_matching_voice(self):
        with _with_settings():
            self.assertEqual(
                PiperTTSEngine.voice_for("female, young adult", "Éva"),
                "hu_HU-berta-medium",
            )
            self.assertEqual(
                PiperTTSEngine.voice_for("male, elderly, low pitch", "Gergely"),
                "hu_HU-imre-medium",
            )

    def test_hungarian_gender_words_are_understood_too(self):
        with _with_settings():
            self.assertEqual(
                PiperTTSEngine.voice_for("férfi, idős", "Gergely"), "hu_HU-imre-medium"
            )

    def test_gender_matching_can_be_turned_off(self):
        with _with_settings(piper_match_gender=False):
            voices = {
                PiperTTSEngine.voice_for("female, young adult", name)
                for name in ("Éva", "Vicuska", "Anna", "Sára", "Ilona")
            }
        # Without matching, female characters land on the male voice as well.
        self.assertIn("hu_HU-imre-medium", voices)

    def test_casting_is_keyed_on_the_character_not_the_description(self):
        with _with_settings():
            first = PiperTTSEngine.voice_for("male, elderly, low pitch", "Gergely")
            edited = PiperTTSEngine.voice_for("male, middle-aged, high pitch", "Gergely")
        self.assertEqual(first, edited)

    def test_casting_is_stable_across_processes(self):
        # hash() is randomized per process; md5 is what keeps a re-export from
        # re-rendering the book in different voices.
        with _with_settings():
            self.assertEqual(
                PiperTTSEngine.voice_for("female", "Éva"), "hu_HU-berta-medium"
            )

    def test_a_single_configured_voice_falls_back_to_the_narrator(self):
        with _with_settings(piper_character_voices=DEFAULT_NARRATOR_VOICE):
            self.assertEqual(
                PiperTTSEngine.voice_for("female", "Éva"), DEFAULT_NARRATOR_VOICE
            )


class TextPreparationTests(unittest.TestCase):
    def setUp(self):
        self.engine = PiperTTSEngine()

    def _prepare(self, text, language="hu"):
        with patch("core.piper_engine.apply_text_normalization",
                   side_effect=lambda t, _: t):
            return self.engine._prepare_text(text, language, True, None)

    def test_enrichment_tags_are_removed_not_spoken(self):
        # espeak turns "[laughter]" into the word "lˈɑuɡhtɛr".
        self.assertEqual(self._prepare("[laughter] Szia. [sigh]"), "Szia.")

    def test_legacy_pdf_vowels_are_repaired(self):
        self.assertEqual(self._prepare("A bûnözõ õrzi."), "A bűnöző őrzi.")

    def test_case_and_punctuation_survive_untouched(self):
        # Unlike F5, espeak handles arbitrary text, so nothing is folded away.
        self.assertEqual(
            self._prepare("A GRÓF ÚR (mérve): 100%-ban!"),
            "A GRÓF ÚR (mérve): 100%-ban!",
        )


class ResampleTests(unittest.TestCase):
    def test_output_is_converted_to_the_pipeline_rate(self):
        # exporter writes merged chapters at a fixed 24 kHz and ignores the rate
        # it read, so 22.05 kHz audio would export ~9% fast.
        audio = np.zeros(22_050, dtype=np.float32)
        converted = PiperTTSEngine._to_pipeline_rate(audio, 22_050)
        self.assertAlmostEqual(len(converted) / 24_000, 1.0, places=2)

    def test_matching_rate_is_passed_through_untouched(self):
        audio = np.zeros(2_400, dtype=np.float32)
        self.assertEqual(
            len(PiperTTSEngine._to_pipeline_rate(audio, 24_000)), len(audio)
        )

    def test_empty_audio_does_not_reach_the_resampler(self):
        self.assertEqual(len(PiperTTSEngine._to_pipeline_rate(np.zeros(0), 22_050)), 0)


class CacheKeyTests(unittest.TestCase):
    def test_key_follows_the_cast_voice(self):
        with _with_settings():
            narration = PiperTTSEngine.cache_key("Szia", "female", None, 1.0)
            character = PiperTTSEngine.cache_key(
                "Szia", "female", None, 1.0, speaker="Éva"
            )
        self.assertNotEqual(narration, character)

    def test_key_ignores_the_reference_clip_because_piper_cannot_clone(self):
        with _with_settings():
            without = PiperTTSEngine.cache_key("Szia", None, None, 1.0)
            with_ref = PiperTTSEngine.cache_key("Szia", None, "ref.wav", 1.0)
        self.assertEqual(without, with_ref)


class RouterTests(unittest.TestCase):
    def test_router_builds_the_piper_engine(self):
        self.assertIn("piper", ENGINE_NAMES)
        with patch("core.tts_router.selected_engine_name", return_value="piper"):
            router = TTSEngineRouter()
        self.assertEqual(router.engine_name, "piper")
        self.assertIsInstance(router._engine, PiperTTSEngine)

    def test_piper_does_not_use_the_narrator_voice_lock(self):
        self.assertFalse(engine_uses_voice_lock("piper"))


class PreviewSignatureTests(unittest.TestCase):
    def test_every_engine_accepts_the_shared_speaker_argument(self):
        # app.py passes speaker= for character previews regardless of engine.
        import inspect

        from core.elevenlabs_engine import ElevenLabsTTSEngine
        from core.f5_engine import F5TTSEngine
        from core.higgs_engine import HiggsTTSEngine
        from core.tts_engine import TTSEngine

        for cls in (TTSEngine, HiggsTTSEngine, F5TTSEngine, ElevenLabsTTSEngine,
                    PiperTTSEngine):
            params = inspect.signature(cls.generate_preview).parameters
            self.assertIn("speaker", params, cls.__name__)


if __name__ == "__main__":
    unittest.main()
