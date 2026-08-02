import os
import tempfile
import unittest
from pathlib import Path

import app as app_module
from core import database, settings


class SettingsTtsCacheInvalidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        self.original_settings_file = settings.SETTINGS_FILE
        self.original_startup = app_module._startup_complete
        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        settings.SETTINGS_FILE = Path(self.tmp.name) / 'settings.json'
        app_module._startup_complete = True
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type) "
                "VALUES (1, 'Test', 'test.txt', 'txt')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'Chapter 1', 0, 'Test text.')"
            )
            conn.execute(
                "INSERT INTO tts_segments "
                "(book_id, chapter_id, segment_index, text, enriched_text, cache_key, audio_path) "
                "VALUES (1, 1, 0, 'Test text.', 'Test text.', 'old-key', 'old.wav')"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        app_module._startup_complete = self.original_startup
        self.tmp.cleanup()

    def _segment_count(self):
        with database.get_conn() as conn:
            return conn.execute('SELECT COUNT(*) FROM tts_segments').fetchone()[0]

    def test_quality_change_clears_persisted_playback_segments(self):
        response = self.client.post('/api/settings', json={'tts_num_step': 32})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 0)

    def test_unchanged_quality_keeps_persisted_playback_segments(self):
        response = self.client.post('/api/settings', json={'tts_num_step': 16})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 1)

    def test_pronunciation_change_clears_persisted_playback_segments(self):
        response = self.client.post(
            '/api/settings',
            json={'pronunciation_dict': 'Westeros = Veszterosz'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()['settings']['pronunciation_dict'],
            'Westeros = Veszterosz',
        )
        self.assertEqual(self._segment_count(), 0)

    def test_voice_lock_change_clears_persisted_playback_segments(self):
        response = self.client.post('/api/settings', json={'narrator_voice_lock': False})

        self.assertEqual(response.status_code, 200)
        self.assertIs(response.get_json()['settings']['narrator_voice_lock'], False)
        self.assertEqual(self._segment_count(), 0)

    def test_unchanged_pronunciation_keeps_persisted_playback_segments(self):
        response = self.client.post('/api/settings', json={'pronunciation_dict': ''})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 1)

    def test_expression_policy_migration_marks_old_prompts_stale_once(self):
        settings.save({'tts_expression_policy_version': 1})

        self.assertTrue(settings.migrate_tts_expression_policy_version())
        self.assertEqual(
            settings.load()['tts_expression_policy_version'],
            settings.TTS_EXPRESSION_POLICY_VERSION,
        )
        self.assertFalse(settings.migrate_tts_expression_policy_version())

    def test_segment_boundary_migration_disables_unsafe_line_merging_once(self):
        settings.save({
            'tts_segment_boundary_policy_version': 0,
            'tts_coalesce_chars': 720,
        })

        self.assertTrue(
            settings.migrate_tts_segment_boundary_policy_version()
        )
        current = settings.load()
        self.assertEqual(current['tts_coalesce_chars'], 0)
        self.assertEqual(
            current['tts_segment_boundary_policy_version'],
            settings.TTS_SEGMENT_BOUNDARY_POLICY_VERSION,
        )
        self.assertFalse(
            settings.migrate_tts_segment_boundary_policy_version()
        )

    def test_settings_api_keeps_exact_line_boundaries_enabled(self):
        response = self.client.post(
            '/api/settings',
            json={'tts_coalesce_chars': 1000},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['settings']['tts_coalesce_chars'], 0)


if __name__ == '__main__':
    unittest.main()
