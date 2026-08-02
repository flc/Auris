"""Clearing the audio cache.

Two things live under audio_cache/ and they are not interchangeable. Rendered
sentences are disposable — deleting them costs time, not identity. The locked
voice samples *are* the identity of an instruction-only narrator, so they only
go when explicitly asked for. These tests exist mostly to keep that line from
being erased by a later refactor.
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


class AudioCacheClearTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original_db_path = database.DB_PATH
        self.original_settings_file = settings.SETTINGS_FILE
        self.original_startup = app_module._startup_complete
        self.originals = (
            tts_engine.AUDIO_CACHE_DIR,
            tts_engine.VOICE_REF_DIR,
            tts_engine.VOICE_PROMPT_DIR,
        )

        root = Path(self.tmp.name)
        self.audio_dir = root / 'audio_cache'
        self.ref_dir = self.audio_dir / 'voice_refs'
        self.prompt_dir = self.audio_dir / 'voice_prompts'
        for directory in (self.audio_dir, self.ref_dir, self.prompt_dir):
            directory.mkdir(parents=True)
        tts_engine.AUDIO_CACHE_DIR = str(self.audio_dir)
        tts_engine.VOICE_REF_DIR = str(self.ref_dir)
        tts_engine.VOICE_PROMPT_DIR = str(self.prompt_dir)

        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        settings.SETTINGS_FILE = root / 'settings.json'
        app_module._startup_complete = True
        database.init_db()

        for name in ('a', 'b', 'c'):
            sf.write(str(self.audio_dir / f'{name}.wav'),
                     np.zeros(24_000, dtype=np.float32), 24_000)
        sf.write(str(self.ref_dir / 'narrator.wav'),
                 np.zeros(24_000, dtype=np.float32), 24_000)
        (self.prompt_dir / 'prompt.pt').write_bytes(b'x' * 1024)

        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type) "
                "VALUES (1, 'T', 't.txt', 'txt')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'C', 0, 'x')"
            )
            conn.execute(
                "INSERT INTO tts_segments "
                "(id, book_id, chapter_id, segment_index, text, enriched_text, "
                " audio_path, duration_sec, cache_key, selected_variant) "
                "VALUES (9, 1, 1, 0, 'x', 'x', ?, 1.0, 'a', 2)",
                (str(self.audio_dir / 'a.wav'),),
            )
            conn.execute(
                "INSERT INTO tts_segment_variants "
                "(segment_id, variant, cache_key, audio_path, duration_sec) "
                "VALUES (9, 2, 'a', ?, 1.0)",
                (str(self.audio_dir / 'a.wav'),),
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        app_module._startup_complete = self.original_startup
        (tts_engine.AUDIO_CACHE_DIR,
         tts_engine.VOICE_REF_DIR,
         tts_engine.VOICE_PROMPT_DIR) = self.originals

    def _names(self, directory):
        return sorted(os.listdir(directory))

    def test_usage_separates_the_two_kinds(self):
        usage = self.client.get('/api/settings/audio-cache').get_json()

        self.assertEqual(usage['rendered_files'], 3)
        self.assertEqual(usage['voice_files'], 2)   # one ref wav + one prompt
        self.assertGreater(usage['rendered_bytes'], usage['voice_bytes'])

    def test_clearing_leaves_the_voice_samples_alone(self):
        """Deleting these would silently re-design the narrator."""
        response = self.client.delete('/api/settings/audio-cache')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['removed_files'], 3)
        self.assertEqual(self._names(self.audio_dir),
                         ['voice_prompts', 'voice_refs'])
        self.assertEqual(self._names(self.ref_dir), ['narrator.wav'])
        self.assertEqual(self._names(self.prompt_dir), ['prompt.pt'])

    def test_voice_samples_go_only_when_asked(self):
        response = self.client.delete(
            '/api/settings/audio-cache?include_voices=1'
        )

        self.assertEqual(response.get_json()['removed_voice_files'], 2)
        self.assertEqual(self._names(self.ref_dir), [])
        self.assertEqual(self._names(self.prompt_dir), [])

    def test_segments_stop_pointing_at_deleted_audio(self):
        self.client.delete('/api/settings/audio-cache')

        with database.get_conn() as conn:
            row = dict(conn.execute('SELECT * FROM tts_segments WHERE id=9').fetchone())
            takes = conn.execute(
                'SELECT COUNT(*) FROM tts_segment_variants'
            ).fetchone()[0]

        self.assertIsNone(row['audio_path'])
        self.assertIsNone(row['cache_key'])
        self.assertEqual(row['selected_variant'], 0)
        self.assertEqual(takes, 0)

    def test_a_running_export_blocks_the_delete(self):
        """The export is reading these files right now."""
        with mock.patch.object(app_module, '_export_exclusive_active', return_value=True):
            response = self.client.delete('/api/settings/audio-cache')

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.get_json()['export_busy'])
        self.assertEqual(len(self._names(self.audio_dir)), 5)

    def test_only_this_app_s_own_files_are_touched(self):
        (self.audio_dir / 'notes.txt').write_text('keep me')
        (self.audio_dir / 'subdir').mkdir()

        self.client.delete('/api/settings/audio-cache?include_voices=1')

        self.assertIn('notes.txt', self._names(self.audio_dir))
        self.assertIn('subdir', self._names(self.audio_dir))

    def test_clearing_an_already_empty_cache_is_harmless(self):
        self.client.delete('/api/settings/audio-cache')

        response = self.client.delete('/api/settings/audio-cache')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['removed_files'], 0)
        self.assertEqual(response.get_json()['freed_bytes'], 0)

    def test_the_reported_saving_matches_what_went_away(self):
        before = self.client.get('/api/settings/audio-cache').get_json()

        freed = self.client.delete('/api/settings/audio-cache').get_json()

        self.assertEqual(freed['freed_bytes'], before['rendered_bytes'])
        self.assertEqual(freed['usage']['rendered_files'], 0)
        self.assertEqual(freed['usage']['voice_files'], before['voice_files'])


if __name__ == '__main__':
    unittest.main()
