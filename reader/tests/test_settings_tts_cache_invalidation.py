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

    def test_voice_lock_change_is_inert_on_an_engine_that_ignores_it(self):
        """ElevenLabs addresses a fixed voice id — a re-render would just re-bill."""
        settings.save({'tts_engine': 'elevenlabs'})

        response = self.client.post('/api/settings', json={'narrator_voice_lock': False})

        self.assertEqual(response.status_code, 200)
        self.assertIs(response.get_json()['settings']['narrator_voice_lock'], False)
        self.assertEqual(self._segment_count(), 1)

    def test_voice_lock_change_is_inert_on_f5(self):
        """F5 clones from a mandatory reference clip; there is no designed voice."""
        settings.save({'tts_engine': 'f5'})

        response = self.client.post('/api/settings', json={'narrator_voice_lock': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 1)

    def test_switching_to_an_engine_that_ignores_voice_lock_still_clears(self):
        """The engine change itself is what invalidates, not the voice lock."""
        response = self.client.post(
            '/api/settings',
            json={'tts_engine': 'elevenlabs', 'narrator_voice_lock': False},
        )

        self.assertEqual(response.status_code, 200)
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

    def test_elevenlabs_voice_change_clears_persisted_playback_segments(self):
        response = self.client.post(
            '/api/settings', json={'elevenlabs_voice_id': ' voice-2 '}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()['settings']['elevenlabs_voice_id'], 'voice-2'
        )
        self.assertEqual(self._segment_count(), 0)

    def test_piper_voice_change_clears_persisted_playback_segments(self):
        """Recasting changes who speaks each line, so old takes are stale."""
        response = self.client.post(
            '/api/settings', json={'piper_narrator_voice': 'hu_HU-imre-medium'}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()['settings']['piper_narrator_voice'],
            'hu_HU-imre-medium',
        )
        self.assertEqual(self._segment_count(), 0)

    def test_piper_character_recast_clears_persisted_playback_segments(self):
        response = self.client.post(
            '/api/settings', json={'piper_match_gender': False}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 0)

    def test_malformed_piper_voice_name_falls_back_instead_of_saving(self):
        """An invalid name would only surface as a 404 mid-chapter."""
        response = self.client.post(
            '/api/settings',
            json={
                'piper_narrator_voice': 'nonsense',
                'piper_character_voices': 'hu_HU-imre-medium, bad name ,',
            },
        )

        saved = response.get_json()['settings']
        self.assertEqual(saved['piper_narrator_voice'], 'hu_HU-anna-medium')
        self.assertEqual(saved['piper_character_voices'], 'hu_HU-imre-medium')

    def test_f5_reference_change_clears_persisted_playback_segments(self):
        response = self.client.post(
            '/api/settings', json={'f5_ref_text': 'Egy másik referencia.'}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 0)

    def test_elevenlabs_api_key_change_keeps_persisted_playback_segments(self):
        """Credentials do not change how a segment sounds."""
        response = self.client.post('/api/settings', json={'elevenlabs_api_key': 'sk_new'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._segment_count(), 1)

    def test_unsupported_elevenlabs_output_format_falls_back(self):
        response = self.client.post(
            '/api/settings', json={'elevenlabs_output_format': 'ulaw_8000'}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()['settings']['elevenlabs_output_format'],
            'mp3_44100_128',
        )

    def test_elevenlabs_voice_settings_are_clamped(self):
        response = self.client.post(
            '/api/settings',
            json={'elevenlabs_stability': 5, 'elevenlabs_style': -2},
        )

        saved = response.get_json()['settings']
        self.assertEqual(saved['elevenlabs_stability'], 1.0)
        self.assertEqual(saved['elevenlabs_style'], 0.0)

    def test_unknown_engine_name_falls_back_to_omnivoice(self):
        response = self.client.post('/api/settings', json={'tts_engine': 'elevenlabz'})

        self.assertEqual(response.get_json()['settings']['tts_engine'], 'omnivoice')

    def test_elevenlabs_engine_can_be_selected(self):
        response = self.client.post('/api/settings', json={'tts_engine': 'ElevenLabs'})

        self.assertEqual(response.get_json()['settings']['tts_engine'], 'elevenlabs')

    def test_settings_api_keeps_exact_line_boundaries_enabled(self):
        response = self.client.post(
            '/api/settings',
            json={'tts_coalesce_chars': 1000},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['settings']['tts_coalesce_chars'], 0)


if __name__ == '__main__':
    unittest.main()
