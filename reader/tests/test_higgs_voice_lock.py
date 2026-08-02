import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from core import tts_engine
from core.higgs_engine import HIGGS_VOICE_LOCK_SEED, HiggsTTSEngine


class _FakeHiggs(HiggsTTSEngine):
    """Higgs engine with the worker RPC replaced by a recorder."""

    def __init__(self):
        super().__init__()
        self._ready = True
        self._sample_rate = 24_000
        self._resolved_model = 'test/model'
        self.calls = []

    def _synthesize(self, text, instruct, ref_audio, ref_text, speed,
                    language, normalize_text, seed=None, lexicon=None):
        self.calls.append({
            'text': text,
            'instruct': instruct,
            'ref_audio': ref_audio,
            'ref_text': ref_text,
            'language': language,
            'seed': seed,
            'lexicon': lexicon,
        })
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.2, 4096).astype(np.float32)


class HiggsVoiceLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for name in ('AUDIO_CACHE_DIR', 'VOICE_REF_DIR'):
            path = os.path.join(self._tmp.name, name.lower())
            os.makedirs(path, exist_ok=True)
            patcher = mock.patch.object(tts_engine, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        # HiggsTTSEngine writes its audio next to the OmniVoice cache.
        patcher = mock.patch('core.higgs_engine.AUDIO_CACHE_DIR', tts_engine.AUDIO_CACHE_DIR)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.engine = _FakeHiggs()

    def test_every_segment_clones_the_same_locked_clip(self):
        self.engine.generate('Első.', instruct='male, elderly', language='hu')
        self.engine.generate('Második.', instruct='male, elderly', language='hu')

        refs = [c['ref_audio'] for c in self.engine.calls]
        self.assertIsNone(refs[0], 'the locked clip itself is rendered without a reference')
        self.assertTrue(all(r == refs[1] for r in refs[1:]), refs)
        self.assertEqual(len(os.listdir(tts_engine.VOICE_REF_DIR)), 1)

    def test_locked_clip_uses_a_reproducible_seed(self):
        self.engine.generate('Szöveg.', instruct='male, elderly', language='hu')
        seed = self.engine.calls[0]['seed']

        self.assertIsNotNone(seed)
        self.assertEqual(seed, self.engine._voice_lock_seed('male, elderly', 'hu'))
        self.assertGreaterEqual(seed, HIGGS_VOICE_LOCK_SEED % 2_147_483_647)
        # Segments keep whatever seed the settings define.
        self.assertIsNone(self.engine.calls[1]['seed'])

    def test_a_different_instruct_picks_a_different_voice(self):
        self.assertNotEqual(
            self.engine._voice_lock_seed('male, elderly', 'hu'),
            self.engine._voice_lock_seed('female, young adult', 'hu'),
        )
        self.assertNotEqual(
            self.engine._voice_ref_path('male, elderly', 'hu'),
            self.engine._voice_ref_path('female, young adult', 'hu'),
        )

    def test_locked_clip_is_rendered_once(self):
        self.engine.generate('Egy.', instruct='male, elderly', language='hu')
        rendered = len(self.engine.calls)
        self.engine.generate('Kettő.', instruct='male, elderly', language='hu')

        self.assertEqual(len(self.engine.calls) - rendered, 1)

    def test_supplied_reference_audio_wins(self):
        ref = os.path.join(self._tmp.name, 'narrator.wav')
        with open(ref, 'wb'):
            pass

        self.engine.generate('Szöveg.', instruct='male, elderly', ref_audio=ref, ref_text='Minta.')

        self.assertEqual(self.engine.calls[-1]['ref_audio'], ref)
        self.assertEqual(os.listdir(tts_engine.VOICE_REF_DIR), [])

    def test_lock_can_be_switched_off(self):
        with mock.patch('core.higgs_engine._voice_lock_enabled', return_value=False):
            self.engine.generate('Szöveg.', instruct='male, elderly', language='hu')

        self.assertIsNone(self.engine.calls[-1]['ref_audio'])
        self.assertEqual(os.listdir(tts_engine.VOICE_REF_DIR), [])

    def test_failed_lock_falls_back_to_plain_generation(self):
        with mock.patch.object(_FakeHiggs, '_synthesize', side_effect=[
            np.zeros(0, dtype=np.float32),          # unusable narrator clip
            np.ones(1024, dtype=np.float32) * 0.1,  # the segment itself
        ]):
            result = self.engine.generate('Szöveg.', instruct='male, elderly', language='hu')

        self.assertTrue(os.path.exists(result['audio_path']))
        self.assertEqual(os.listdir(tts_engine.VOICE_REF_DIR), [])

    def test_language_selects_the_reference_sentence(self):
        self.engine.generate('Szöveg.', instruct='male, elderly', language='hu')
        self.assertEqual(self.engine.calls[0]['text'], tts_engine.VOICE_DESIGN_REF_TEXTS['hu'])

        self.engine.generate('Text.', instruct='male, elderly', language='en')
        self.assertEqual(self.engine.calls[2]['text'], tts_engine.VOICE_DESIGN_REF_TEXT)
        self.assertEqual(len(os.listdir(tts_engine.VOICE_REF_DIR)), 2)


if __name__ == '__main__':
    unittest.main()
