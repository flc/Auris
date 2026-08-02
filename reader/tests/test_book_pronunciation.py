import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app as app_module
from core import database, settings
from core.pronunciation import apply_pronunciation, combine_lexicons


class CombineLexiconsTest(unittest.TestCase):
    def test_earlier_source_wins_on_the_same_word(self):
        merged = combine_lexicons('Aegon = Egon', 'Aegon = Ígon\nWesteros = Veszterosz')

        self.assertEqual(apply_pronunciation('Aegon', merged), 'Egon')
        # Rules the book does not mention are still inherited.
        self.assertEqual(apply_pronunciation('Westeros', merged), 'Veszterosz')

    def test_empty_sources_are_dropped(self):
        self.assertEqual(combine_lexicons('', None, 'Aegon = Egon'), 'Aegon = Egon')
        self.assertEqual(combine_lexicons(None, ''), '')


class _AppTestCase(unittest.TestCase):
    """Flask app wired to a throwaway database and settings file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original_db_path = database.DB_PATH
        self.original_settings_file = settings.SETTINGS_FILE
        self.original_startup = app_module._startup_complete
        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        settings.SETTINGS_FILE = Path(self.tmp.name) / 'settings.json'
        app_module._startup_complete = True
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type, language, narrator_instruct) "
                "VALUES (1, 'Test', 'test.txt', 'txt', 'hu', 'male, elderly')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'Chapter 1', 0, 'Test text.')"
            )
            conn.execute(
                "INSERT INTO tts_segments "
                "(book_id, chapter_id, segment_index, text, enriched_text, cache_key, audio_path) "
                "VALUES (1, 1, 0, 'Test.', 'Test.', 'old-key', 'old.wav')"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        app_module._startup_complete = self.original_startup

    def _segment_count(self):
        with database.get_conn() as conn:
            return conn.execute('SELECT COUNT(*) FROM tts_segments').fetchone()[0]

class BookPronunciationTest(_AppTestCase):
    def test_books_table_has_a_pronunciation_column(self):
        with database.get_conn() as conn:
            cols = {row[1] for row in conn.execute('PRAGMA table_info(books)')}

        self.assertIn('pronunciation_dict', cols)

    def test_book_rules_extend_and_override_the_global_ones(self):
        settings.save({'pronunciation_dict': 'Aegon = Ígon\nWesteros = Veszterosz'})
        with database.get_conn() as conn:
            conn.execute("UPDATE books SET pronunciation_dict='Aegon = Egon' WHERE id=1")

        merged = app_module._book_pronunciation(1)

        self.assertEqual(apply_pronunciation('Aegon', merged), 'Egon')
        self.assertEqual(apply_pronunciation('Westeros', merged), 'Veszterosz')

    def test_book_without_rules_falls_back_to_the_global_lexicon(self):
        settings.save({'pronunciation_dict': 'Westeros = Veszterosz'})

        merged = app_module._book_pronunciation(1)

        self.assertEqual(apply_pronunciation('Westerosban', merged), 'Veszteroszban')

    def test_narrator_api_round_trips_the_book_lexicon(self):
        response = self.client.put(
            '/api/books/1/narrator',
            json={'instruct': 'male, elderly', 'pronunciation_dict': 'Aegon = Egon'},
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['pronunciation_dict'], 'Aegon = Egon')
        self.assertTrue(body['segments_cleared'])
        self.assertEqual(self._segment_count(), 0)

        stored = self.client.get('/api/books/1/narrator').get_json()
        self.assertEqual(stored['pronunciation_dict'], 'Aegon = Egon')

    def test_reader_endpoint_returns_the_merged_rules_and_matcher(self):
        settings.save({'pronunciation_dict': 'Westeros = Veszterosz'})
        with database.get_conn() as conn:
            conn.execute("UPDATE books SET pronunciation_dict='Aegon = Egon' WHERE id=1")

        body = self.client.get('/api/books/1/pronunciation').get_json()

        self.assertEqual(
            body['entries'],
            [
                {'source': 'Aegon', 'spoken': 'Egon'},
                {'source': 'Westeros', 'spoken': 'Veszterosz'},
            ],
        )
        self.assertIn('Westeros', body['pattern'])
        self.assertIn('Aegon', body['pattern'])

    def test_reader_endpoint_404s_for_an_unknown_book(self):
        self.assertEqual(self.client.get('/api/books/999/pronunciation').status_code, 404)

    def test_unchanged_lexicon_keeps_persisted_segments(self):
        response = self.client.put(
            '/api/books/1/narrator',
            json={'instruct': 'male, elderly', 'pronunciation_dict': ''},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()['segments_cleared'])
        self.assertEqual(self._segment_count(), 1)


class DefaultNarratorPreviewTest(_AppTestCase):
    def test_preview_requires_a_loaded_model(self):
        with mock.patch.object(app_module.tts, 'status', return_value={'state': 'not_loaded'}):
            response = self.client.post('/api/settings/narrator-preview', json={})

        self.assertEqual(response.status_code, 503)

    def test_preview_uses_the_saved_instruct_and_language_sample(self):
        calls = {}

        def _fake_preview(**kwargs):
            calls.update(kwargs)
            return {'cache_key': 'abc123'}

        settings.save({'narrator_instruct': 'female, young adult'})
        with mock.patch.object(app_module.tts, 'status', return_value={'state': 'ready'}), \
             mock.patch.object(app_module.tts, 'generate_preview', side_effect=_fake_preview):
            response = self.client.post('/api/settings/narrator-preview', json={'language': 'hu'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['audio_url'], '/api/audio/abc123')
        self.assertEqual(calls['instruct'], 'female, young adult')
        self.assertEqual(calls['language'], 'hu')
        self.assertEqual(calls['sample_text'], app_module.VOICE_PREVIEW_TEXTS['hu'])

    def test_blank_instruct_falls_back_to_the_saved_default(self):
        calls = {}

        def _fake_preview(**kwargs):
            calls.update(kwargs)
            return {'cache_key': 'abc123'}

        # settings.load() backfills a blank instruct with the shipped default,
        # so a blank request still previews the voice books actually get.
        settings.save({'narrator_instruct': ''})
        with mock.patch.object(app_module.tts, 'status', return_value={'state': 'ready'}), \
             mock.patch.object(app_module.tts, 'generate_preview', side_effect=_fake_preview):
            response = self.client.post('/api/settings/narrator-preview', json={'instruct': '  '})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls['instruct'], settings.DEFAULT_NARRATOR_INSTRUCT)
        self.assertEqual(calls['sample_text'], app_module.VOICE_PREVIEW_TEXT)


if __name__ == '__main__':
    unittest.main()
