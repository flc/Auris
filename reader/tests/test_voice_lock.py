import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from core import tts_engine
from core.tts_engine import TTSEngine


class _FakeModel:
    """Stands in for OmniVoice: records every generate() call."""

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs.get("text")
        count = len(text) if isinstance(text, list) else 1
        # Noisy enough to pass the voice-reference stability check.
        rng = np.random.default_rng(0)
        return [rng.normal(0, 0.2, 4096).astype(np.float32) for _ in range(count)]


class VoiceLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patches = {
            "AUDIO_CACHE_DIR": os.path.join(self._tmp.name, "audio_cache"),
            "VOICE_REF_DIR": os.path.join(self._tmp.name, "voice_refs"),
            "VOICE_PROMPT_DIR": os.path.join(self._tmp.name, "voice_prompts"),
        }
        for name, path in patches.items():
            os.makedirs(path, exist_ok=True)
            self.addCleanup(mock.patch.object(tts_engine, name, path).stop)
            mock.patch.object(tts_engine, name, path).start()

        self.engine = TTSEngine()
        self.engine.model = _FakeModel()
        self.engine._ready = True

    def _generate(self, text, **kwargs):
        return self.engine.generate(text, num_step=8, normalize_text=False, **kwargs)

    def test_locked_narrator_reuses_one_reference_for_every_segment(self):
        self._generate("Első szegmens.", instruct="male, elderly, low pitch", language="hu")
        self._generate("Második szegmens.", instruct="male, elderly, low pitch", language="hu")

        # First call renders the reference clip, then each segment clones it.
        refs = [c.get("ref_audio") or c.get("voice_clone_prompt") for c in self.engine.model.calls]
        self.assertIsNone(refs[0], "the reference clip itself is voice-designed")
        self.assertTrue(all(r is not None for r in refs[1:]), refs)

        instructs = [c.get("instruct") for c in self.engine.model.calls[1:]]
        self.assertEqual(instructs, [None, None], "segments must not re-run voice design")

        stored = os.listdir(tts_engine.VOICE_REF_DIR)
        self.assertEqual(len(stored), 1, stored)

    def test_reference_is_rendered_once_and_then_cached(self):
        self._generate("Egy.", instruct="male, elderly", language="hu")
        first = len(self.engine.model.calls)
        self._generate("Kettő.", instruct="male, elderly", language="hu")

        # Second segment: one generate() call, no new reference rendering.
        self.assertEqual(len(self.engine.model.calls) - first, 1)

    def test_supplied_reference_audio_still_wins(self):
        ref = os.path.join(self._tmp.name, "narrator.wav")
        with open(ref, "wb"):
            pass

        self._generate("Szöveg.", instruct="male, elderly", ref_audio=ref, ref_text="Minta.")

        call = self.engine.model.calls[-1]
        self.assertEqual(call.get("ref_audio"), ref)
        self.assertIsNone(call.get("instruct"))
        self.assertEqual(os.listdir(tts_engine.VOICE_REF_DIR), [])

    def test_lock_can_be_switched_off(self):
        with mock.patch.object(tts_engine, "_voice_lock_enabled", return_value=False):
            self._generate("Szöveg.", instruct="male, elderly", language="hu")

        call = self.engine.model.calls[-1]
        self.assertEqual(call.get("instruct"), "male, elderly")
        self.assertEqual(os.listdir(tts_engine.VOICE_REF_DIR), [])

    def test_reference_sample_uses_the_book_language(self):
        self._generate("Szöveg.", instruct="male, elderly", language="hu")
        hu_ref_text = self.engine.model.calls[0]["text"]
        self.assertEqual(hu_ref_text, tts_engine.VOICE_DESIGN_REF_TEXTS["hu"])

        self._generate("Text.", instruct="male, elderly", language="en")
        en_ref_text = self.engine.model.calls[2]["text"]
        self.assertEqual(en_ref_text, tts_engine.VOICE_DESIGN_REF_TEXT)

        # Different languages must not share one reference clip.
        self.assertEqual(len(os.listdir(tts_engine.VOICE_REF_DIR)), 2)

    def test_each_item_is_rewritten_with_its_own_lexicon(self):
        items = [
            {"text": "Westeros", "instruct": "male, elderly", "language": "hu",
             "lexicon": "Westeros = Veszterosz"},
            {"text": "Westeros", "instruct": "male, elderly", "language": "hu",
             "lexicon": ""},
        ]
        self.engine.generate_many(items, num_step=8, batch_size=4)

        spoken = []
        for call in self.engine.model.calls:
            text = call.get("text")
            spoken.extend(text if isinstance(text, list) else [text])

        # Books with different rules must not be merged into one batch.
        self.assertIn("Veszterosz", spoken)
        self.assertIn("Westeros", spoken)

    def test_batch_path_locks_the_same_voice(self):
        items = [
            {"text": "Egy.", "instruct": "male, elderly", "language": "hu"},
            {"text": "Kettő.", "instruct": "male, elderly", "language": "hu"},
        ]
        self.engine.generate_many(items, num_step=8, batch_size=4)

        stored = os.listdir(tts_engine.VOICE_REF_DIR)
        self.assertEqual(len(stored), 1, stored)
        batch_calls = [c for c in self.engine.model.calls if isinstance(c.get("text"), list)]
        self.assertTrue(batch_calls, "expected one batched generate() call")
        self.assertIsNone(batch_calls[-1].get("instruct"))


if __name__ == "__main__":
    unittest.main()
