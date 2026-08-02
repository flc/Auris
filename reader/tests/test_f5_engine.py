"""Unit tests for the F5-TTS engine.

The interesting logic is all in front of the model: F5-TTS maps any character
outside its 67-token vocabulary to index 0 — a space — so a folding mistake
does not raise, it quietly deletes letters mid-word. These tests pin that down
without loading weights.
"""

import unittest
from unittest.mock import patch

import numpy as np

from core.f5_engine import F5TTSEngine, _trim_onset_artifact
from core.tts_router import TTSEngineRouter

# The vocabulary the Hungarian checkpoints ship: lowercase only, no
# parentheses, no square brackets, no uppercase.
HUNGARIAN_VOCAB = frozenset(
    " !%&*,-.0123456789:;?abcdefghijklmnopqrstuvwxyz»àáéíóöúüčőű–'\"„…"
)


def _engine_with_vocabulary():
    engine = F5TTSEngine()
    engine._vocab_chars = HUNGARIAN_VOCAB
    return engine


class TextFoldingTests(unittest.TestCase):
    def setUp(self):
        self.engine = _engine_with_vocabulary()

    def _fold(self, text, language="hu", normalize=True):
        # apply_text_normalization reaches for OmniVoice/wetext; the num2words
        # fallback is what ships on every platform, so pin the test to it.
        with patch("core.f5_engine.apply_text_normalization", side_effect=lambda t, _: t):
            return self.engine._prepare_text(text, language, normalize, None)

    def test_uppercase_is_folded_rather_than_silently_dropped(self):
        # Every uppercase letter would otherwise become a space.
        self.assertEqual(self._fold("A GRÓF ÚR"), "a gróf úr")

    def test_enrichment_tags_never_reach_the_tokenizer(self):
        folded = self._fold("[laughter] Nevetett. [surprise-wa]")
        self.assertNotIn("[", folded)
        self.assertNotIn("laughter", folded)
        self.assertEqual(folded, "nevetett.")

    def test_parentheses_become_a_single_pause_not_a_stutter(self):
        # "(mérve)," folds to ",mérve,," — three pauses where one belongs.
        self.assertEqual(self._fold("Ez volt (mérve), igen."), "ez volt, mérve, igen.")

    def test_strongest_mark_survives_a_collapsed_punctuation_run(self):
        # The closing parenthesis must not demote the question mark to a comma.
        self.assertEqual(self._fold("Mennyi? (Semmi.)"), "mennyi? semmi.")

    def test_hungarian_symbols_are_spelled_out_with_their_suffix_attached(self):
        self.assertEqual(self._fold("100%-ban"), "100 százalék-ban")
        self.assertEqual(self._fold("2+2"), "2 plusz 2")

    def test_symbols_are_left_alone_for_other_languages(self):
        self.assertEqual(self._fold("100%", language="en"), "100%")

    def test_legacy_pdf_vowels_are_repaired_before_folding(self):
        # 'õ' and 'û' are not in the vocabulary; unrepaired they become spaces.
        self.assertEqual(self._fold("A bûnözõ õrzi a fõbejáratot."),
                         "a bűnöző őrzi a főbejáratot.")

    def test_out_of_vocabulary_characters_do_not_glue_words_together(self):
        self.assertEqual(self._fold("egy → kettő"), "egy kettő")

    def test_ellipsis_is_normalized_to_the_single_vocabulary_glyph(self):
        self.assertEqual(self._fold("Hát ez... érdekes"), "hát ez… érdekes")


class VoiceResolutionTests(unittest.TestCase):
    def test_missing_narrator_reference_fails_loudly(self):
        engine = F5TTSEngine()
        with patch("core.f5_engine._setting", side_effect=lambda key, default: default):
            with self.assertRaises(RuntimeError) as caught:
                engine._resolve_voice("male, elderly", None, None)
        self.assertIn("cannot synthesize a voice from a description",
                      str(caught.exception).lower())

    def test_uploaded_reference_without_a_transcript_is_rejected(self):
        engine = F5TTSEngine()
        with self.assertRaises(RuntimeError) as caught:
            engine._resolve_voice(None, "/tmp/speaker.wav", "   ")
        self.assertIn("transcript", str(caught.exception).lower())

    def test_uploaded_reference_wins_over_the_configured_narrator(self):
        engine = F5TTSEngine()
        with patch("core.f5_engine._setting", side_effect=lambda key, default: {
            "f5_ref_audio": "/tmp/narrator.wav",
            "f5_ref_text": "narrator line",
        }.get(key, default)):
            audio, text = engine._resolve_voice(None, "/tmp/speaker.wav", "speaker line")
        self.assertEqual((audio, text), ("/tmp/speaker.wav", "speaker line"))


class OnsetTrimTests(unittest.TestCase):
    def test_warmup_burst_is_removed_and_speech_is_kept(self):
        sample_rate = 24_000
        burst = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
        burst[:200] = 0.002  # near-silent attention warmup
        speech = np.full(sample_rate, 0.5, dtype=np.float32)
        trimmed = _trim_onset_artifact(np.concatenate([burst, speech]), sample_rate)

        self.assertLess(len(trimmed), len(burst) + len(speech))
        self.assertGreaterEqual(len(trimmed), len(speech))

    def test_clip_shorter_than_the_search_window_is_returned_untouched(self):
        short = np.full(100, 0.5, dtype=np.float32)
        self.assertEqual(len(_trim_onset_artifact(short, 24_000)), 100)


class CacheKeyTests(unittest.TestCase):
    def test_key_changes_with_sampling_settings(self):
        with patch("core.f5_engine.F5TTSEngine._generation_settings",
                   return_value={"nfe_step": 32}):
            first = F5TTSEngine.cache_key("Szia", None, None, 1.0)
        with patch("core.f5_engine.F5TTSEngine._generation_settings",
                   return_value={"nfe_step": 16}):
            second = F5TTSEngine.cache_key("Szia", None, None, 1.0)
        self.assertNotEqual(first, second)

    def test_key_changes_with_the_reference_clip(self):
        first = F5TTSEngine.cache_key("Szia", None, "a.wav", 1.0, ref_text="a")
        second = F5TTSEngine.cache_key("Szia", None, "b.wav", 1.0, ref_text="a")
        self.assertNotEqual(first, second)


class RouterTests(unittest.TestCase):
    def test_router_builds_the_f5_engine(self):
        with patch("core.tts_router.selected_engine_name", return_value="f5"):
            router = TTSEngineRouter()
        self.assertEqual(router.engine_name, "f5")
        self.assertIsInstance(router._engine, F5TTSEngine)


if __name__ == "__main__":
    unittest.main()
