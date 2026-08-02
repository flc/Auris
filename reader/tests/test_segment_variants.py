"""Alternative takes of a single segment.

A generated reading is a sample, not a fact — a model sometimes swallows a word
or lands the stress wrongly. These cover the machinery that lets a listener ask
one sentence for another reading and keep the better one: the takes have to be
genuinely different audio, the original take must keep the cache key it always
had, and whichever take is kept has to be the one playback and export pick up.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import app as app_module
from core import database, settings, tts_engine
from core.elevenlabs_engine import ElevenLabsTTSEngine
from core.f5_engine import F5TTSEngine
from core.higgs_engine import HiggsTTSEngine
from core.piper_engine import PiperTTSEngine
from core.tts_engine import TTSEngine, variant_seed

ENGINES = (TTSEngine, HiggsTTSEngine, F5TTSEngine, PiperTTSEngine, ElevenLabsTTSEngine)


class VariantSeedTests(unittest.TestCase):
    def test_the_original_take_keeps_the_configured_seed(self):
        self.assertEqual(variant_seed(1234, 0), 1234)
        self.assertEqual(variant_seed(-1, 0), -1)

    def test_alternatives_are_distinct_and_repeatable(self):
        seeds = [variant_seed(1234, n) for n in range(1, 6)]

        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(seeds, [variant_seed(1234, n) for n in range(1, 6)])

    def test_a_random_configured_seed_still_yields_a_usable_one(self):
        """-1 means 'pick for me', but an alternative needs a real number."""
        self.assertGreater(variant_seed(-1, 1), 0)


class VariantCacheKeyTests(unittest.TestCase):
    def test_every_engine_separates_takes(self):
        for cls in ENGINES:
            with self.subTest(engine=cls.engine_name):
                keys = {cls.cache_key('Hello', None, None, 1.0, variant=n) for n in range(4)}
                self.assertEqual(len(keys), 4)

    def test_the_original_take_keeps_its_existing_key(self):
        """Otherwise this feature would orphan every WAV already generated."""
        for cls in ENGINES:
            with self.subTest(engine=cls.engine_name):
                self.assertEqual(
                    cls.cache_key('Hello', None, None, 1.0),
                    cls.cache_key('Hello', None, None, 1.0, variant=0),
                )

    def test_every_engine_takes_the_shared_arguments(self):
        import inspect

        for cls in ENGINES:
            with self.subTest(engine=cls.engine_name):
                params = inspect.signature(cls.generate).parameters
                self.assertIn('variant', params)
                self.assertIn('speaker', params)


class _FakeEngine:
    """Records the variants asked for and writes a distinguishable WAV each."""

    engine_name = 'fake'
    supports_variants = True

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.calls = []

    def status(self):
        return {'state': 'ready'}

    def generate(self, **kwargs):
        variant = int(kwargs.get('variant', 0))
        self.calls.append(variant)
        key = f'fake-variant-{variant}'
        path = os.path.join(self.cache_dir, f'{key}.wav')
        # Length encodes the take, so a test can tell which one is in use.
        sf.write(path, np.zeros(2400 * (variant + 1), dtype=np.float32), 24_000)
        return {
            'audio_path': path,
            'duration_sec': 0.1 * (variant + 1),
            'cache_hit': False,
            'cache_key': key,
        }


class SegmentVariantApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original_db_path = database.DB_PATH
        self.original_settings_file = settings.SETTINGS_FILE
        self.original_startup = app_module._startup_complete
        self.original_cache_dir = tts_engine.AUDIO_CACHE_DIR
        self.original_tts = app_module.tts

        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        settings.SETTINGS_FILE = Path(self.tmp.name) / 'settings.json'
        tts_engine.AUDIO_CACHE_DIR = self.tmp.name
        app_module._startup_complete = True
        app_module.tts = _FakeEngine(self.tmp.name)
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type, language) "
                "VALUES (1, 'Test', 't.txt', 'txt', 'hu')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'Chapter 1', 0, 'Egy mondat.')"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

        # Let the app build the segment itself: a hand-written row whose
        # instruct or enrichment differs from what the app would compute gets
        # silently rebuilt on first read, which would move the ground out from
        # under every assertion here.
        self.client.get('/api/tts/segments/1/1')
        sf.write(
            os.path.join(self.tmp.name, 'orig-key.wav'),
            np.zeros(48_000, dtype=np.float32),
            24_000,
        )
        with database.get_conn() as conn:
            # Stand in for the take the listener has already heard.
            conn.execute(
                'UPDATE tts_segments SET audio_path=?, duration_sec=2.0, '
                "cache_key='orig-key' WHERE book_id=1 AND chapter_id=1 AND segment_index=0",
                (os.path.join(self.tmp.name, 'orig-key.wav'),),
            )
            self.segment_id = conn.execute(
                'SELECT id FROM tts_segments WHERE book_id=1 AND chapter_id=1 '
                'AND segment_index=0'
            ).fetchone()['id']

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        tts_engine.AUDIO_CACHE_DIR = self.original_cache_dir
        app_module._startup_complete = self.original_startup
        app_module.tts = self.original_tts

    def _segment_row(self):
        with database.get_conn() as conn:
            return dict(conn.execute(
                'SELECT * FROM tts_segments WHERE id=?', (self.segment_id,)
            ).fetchone())

    def test_listing_starts_empty_but_reports_support(self):
        body = self.client.get('/api/tts/variants/1/1/0').get_json()

        self.assertTrue(body['supported'])
        self.assertEqual(body['variants'], [])
        self.assertEqual(body['selected_variant'], 0)

    def test_generating_registers_the_audio_already_played_as_take_one(self):
        """The take you just heard has to be in the comparison."""
        body = self.client.post('/api/tts/variants/1/1/0', json={'count': 2}).get_json()

        variants = {v['variant'] for v in body['variants']}
        self.assertEqual(variants, {0, 1, 2})
        # Take 0 was adopted from the segment, not synthesized again.
        self.assertEqual(app_module.tts.calls, [1, 2])

    def test_a_second_round_continues_the_numbering(self):
        self.client.post('/api/tts/variants/1/1/0', json={'count': 2})
        body = self.client.post('/api/tts/variants/1/1/0', json={'count': 2}).get_json()

        self.assertEqual({v['variant'] for v in body['variants']}, {0, 1, 2, 3, 4})
        self.assertEqual(app_module.tts.calls, [1, 2, 3, 4])

    def test_the_count_is_capped(self):
        self.client.post('/api/tts/variants/1/1/0', json={'count': 99})

        self.assertLessEqual(len(app_module.tts.calls), app_module.MAX_SEGMENT_VARIANTS)

    def test_keeping_a_take_repoints_the_segment(self):
        self.client.post('/api/tts/variants/1/1/0', json={'count': 2})

        response = self.client.post(
            '/api/tts/variants/1/1/0/select', json={'variant': 2}
        )

        self.assertEqual(response.status_code, 200)
        row = self._segment_row()
        self.assertEqual(row['cache_key'], 'fake-variant-2')
        self.assertEqual(row['selected_variant'], 2)
        self.assertAlmostEqual(row['duration_sec'], 0.3)

    def test_playback_metadata_reports_the_kept_take(self):
        self.client.post('/api/tts/variants/1/1/0', json={'count': 2})
        self.client.post('/api/tts/variants/1/1/0/select', json={'variant': 1})

        segments = self.client.get('/api/tts/segments/1/1').get_json()

        self.assertEqual(segments[0]['selected_variant'], 1)
        self.assertEqual(segments[0]['cache_key'], 'fake-variant-1')

    def test_going_back_to_the_original_take_works(self):
        self.client.post('/api/tts/variants/1/1/0', json={'count': 1})
        self.client.post('/api/tts/variants/1/1/0/select', json={'variant': 1})

        self.client.post('/api/tts/variants/1/1/0/select', json={'variant': 0})

        row = self._segment_row()
        self.assertEqual(row['cache_key'], 'orig-key')
        self.assertEqual(row['selected_variant'], 0)

    def test_an_ungenerated_take_cannot_be_kept(self):
        response = self.client.post('/api/tts/variants/1/1/0/select', json={'variant': 3})

        self.assertEqual(response.status_code, 404)

    def test_an_engine_that_cannot_vary_is_refused_with_a_reason(self):
        app_module.tts.supports_variants = False

        response = self.client.post('/api/tts/variants/1/1/0', json={'count': 2})

        self.assertEqual(response.status_code, 400)
        self.assertIn('cannot vary', response.get_json()['error'])
        self.assertEqual(app_module.tts.calls, [])

    def test_an_unloaded_engine_is_refused(self):
        with mock.patch.object(
            app_module.tts, 'status', return_value={'state': 'not_loaded'}
        ):
            response = self.client.post('/api/tts/variants/1/1/0', json={'count': 2})

        self.assertEqual(response.status_code, 503)

    def test_a_running_export_holds_the_engine(self):
        with mock.patch.object(app_module, '_export_exclusive_active', return_value=True):
            response = self.client.post('/api/tts/variants/1/1/0', json={'count': 2})

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.get_json()['export_busy'])

    def test_an_unknown_segment_is_404(self):
        self.assertEqual(self.client.get('/api/tts/variants/1/1/99').status_code, 404)
        self.assertEqual(
            self.client.post('/api/tts/variants/1/1/99', json={}).status_code, 404
        )

    def test_takes_die_with_the_segment_they_belong_to(self):
        """Re-enriching a chapter drops segments; orphan takes must not linger."""
        self.client.post('/api/tts/variants/1/1/0', json={'count': 2})

        with database.get_conn() as conn:
            conn.execute('DELETE FROM tts_segments WHERE id=?', (self.segment_id,))
            remaining = conn.execute(
                'SELECT COUNT(*) FROM tts_segment_variants'
            ).fetchone()[0]

        self.assertEqual(remaining, 0)


class _TruncatingEngine:
    """First take of the marked sentence is short; the retry is complete."""

    engine_name = 'fake'
    supports_variants = True

    def __init__(self, cache_dir, short_text):
        self.cache_dir = cache_dir
        self.short_text = short_text
        self.variants_asked = []

    def status(self):
        return {'state': 'ready'}

    def _render(self, text, variant):
        truncated = self.short_text in text and variant == 0
        seconds = 0.5 if truncated else len(text) / 15.0
        key = f'seg-{abs(hash(text)) % 10000}-v{variant}'
        path = os.path.join(self.cache_dir, f'{key}.wav')
        sf.write(path, np.zeros(int(24_000 * seconds), dtype=np.float32), 24_000)
        return {
            'audio_path': path,
            'duration_sec': seconds,
            'cache_hit': False,
            'cache_key': key,
        }

    def generate(self, **kwargs):
        variant = int(kwargs.get('variant', 0))
        self.variants_asked.append(variant)
        return self._render(kwargs['text'], variant)

    def generate_many(self, items, on_item=None, **kwargs):
        results = []
        for index, item in enumerate(items):
            result = self._render(item['text'], 0)
            results.append(result)
            if on_item:
                on_item(index, result)
        return results


class SuspectRetryTest(unittest.TestCase):
    """The export path repairs sentences that came back too short."""

    SHORT = 'Ez a mondat elharapva jott vissza a modelltol'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original_db_path = database.DB_PATH
        self.original_settings_file = settings.SETTINGS_FILE
        self.original_startup = app_module._startup_complete
        self.original_cache_dir = tts_engine.AUDIO_CACHE_DIR
        self.original_tts = app_module.tts

        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        settings.SETTINGS_FILE = Path(self.tmp.name) / 'settings.json'
        tts_engine.AUDIO_CACHE_DIR = self.tmp.name
        app_module._startup_complete = True
        app_module.tts = _TruncatingEngine(self.tmp.name, self.SHORT)
        database.init_db()

        # Enough normal sentences for the chapter to calibrate its own pace.
        sentences = [
            f'Ez a {n}. rendes hosszusagu mondat a fejezetben, semmi baja nincs.'
            for n in range(1, 11)
        ] + [self.SHORT + ' de ettol meg eleg hosszu ahhoz hogy merni lehessen.']
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type, language) "
                "VALUES (1, 'Test', 't.txt', 'txt', 'hu')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'Chapter 1', 0, ?)",
                (' '.join(sentences),),
            )
        self.addCleanup(self._restore)

    def _restore(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        tts_engine.AUDIO_CACHE_DIR = self.original_cache_dir
        app_module._startup_complete = self.original_startup
        app_module.tts = self.original_tts

    def _generate_chapter(self):
        segs = app_module._get_chapter_segments(1, 1)
        app_module._ensure_audio_for_chapter(1, 1, segs)
        return segs

    def _short_segment(self, segs):
        return next(s for s in segs if self.SHORT in s['text'])

    def test_the_truncated_sentence_is_re_rendered(self):
        segs = self._generate_chapter()
        short = self._short_segment(segs)

        self.assertGreater(short['duration_sec'], 1.0)
        self.assertIn('-v1', short['cache_key'])

    def test_the_repair_is_persisted_for_playback_and_export(self):
        self._generate_chapter()

        with database.get_conn() as conn:
            row = dict(conn.execute(
                'SELECT * FROM tts_segments WHERE text LIKE ?', (f'%{self.SHORT}%',)
            ).fetchone())

        self.assertEqual(row['selected_variant'], 1)
        self.assertIn('-v1', row['cache_key'])

    def test_the_rejected_take_stays_auditionable(self):
        """The listener can still compare against what the model first said."""
        self._generate_chapter()

        with database.get_conn() as conn:
            variants = [dict(r) for r in conn.execute(
                'SELECT v.variant, v.duration_sec FROM tts_segment_variants v '
                'JOIN tts_segments s ON s.id = v.segment_id '
                'WHERE s.text LIKE ? ORDER BY v.variant', (f'%{self.SHORT}%',)
            )]

        self.assertEqual([v['variant'] for v in variants], [0, 1])
        self.assertAlmostEqual(variants[0]['duration_sec'], 0.5)

    def test_healthy_sentences_are_never_re_rendered(self):
        self._generate_chapter()

        # One retry, for the one bad sentence.
        self.assertEqual(app_module.tts.variants_asked, [1])

    def test_the_check_can_be_switched_off(self):
        settings.save({'tts_retry_suspect_segments': False})

        segs = self._generate_chapter()

        self.assertEqual(app_module.tts.variants_asked, [])
        self.assertAlmostEqual(self._short_segment(segs)['duration_sec'], 0.5)

    def test_an_engine_that_cannot_vary_is_left_alone(self):
        """Retrying would hand back the identical file."""
        app_module.tts.supports_variants = False

        self._generate_chapter()

        self.assertEqual(app_module.tts.variants_asked, [])


if __name__ == '__main__':
    unittest.main()
