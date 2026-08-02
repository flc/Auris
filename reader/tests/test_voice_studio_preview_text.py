"""Voice Studio previews can be driven by a custom line.

The point of the field is to hear a voice on the sentence you care about, so
what matters is that a typed line reaches the engine verbatim, that leaving it
alone reproduces exactly the old behaviour, and that it never leaks into
anything that gets saved or exported.
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


class VoiceStudioPreviewTextTest(unittest.TestCase):
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
                "INSERT INTO books (id, title, file_path, file_type, language, "
                "narrator_instruct) "
                "VALUES (1, 'Test', 'test.txt', 'txt', 'hu', 'male, elderly')"
            )
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content) "
                "VALUES (1, 1, 'Chapter 1', 0, 'Test text.')"
            )
            conn.execute(
                "INSERT INTO characters (id, book_id, name, instruct, gender, "
                "frequency, color_hex) "
                "VALUES (7, 1, 'Aegon', 'male, young adult', 'male', 3, '#ffcc00')"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        settings.SETTINGS_FILE = self.original_settings_file
        app_module._startup_complete = self.original_startup

    def _preview(self, url, payload):
        calls = {}

        def fake_preview(**kwargs):
            calls.update(kwargs)
            return {'cache_key': 'abc123'}

        with mock.patch.object(app_module.tts, 'status', return_value={'state': 'ready'}), \
             mock.patch.object(app_module.tts, 'generate_preview', side_effect=fake_preview):
            response = self.client.post(url, json=payload)
        return response, calls

    # ── narrator ─────────────────────────────────────────────────────────────

    def test_narrator_preview_speaks_the_typed_line(self):
        response, calls = self._preview(
            '/api/books/1/characters/narrator/preview',
            {'sample_text': '  Sárkánykő felett vörös az ég.  '},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls['sample_text'], 'Sárkánykő felett vörös az ég.')

    def test_narrator_preview_without_a_line_keeps_the_language_default(self):
        response, calls = self._preview(
            '/api/books/1/characters/narrator/preview', {'instruct': 'male, elderly'}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls['sample_text'], app_module.VOICE_PREVIEW_TEXTS['hu'])

    def test_a_cleared_narrator_field_falls_back_to_the_default(self):
        _, calls = self._preview(
            '/api/books/1/characters/narrator/preview', {'sample_text': '   '}
        )

        self.assertEqual(calls['sample_text'], app_module.VOICE_PREVIEW_TEXTS['hu'])

    # ── characters ───────────────────────────────────────────────────────────

    def test_character_preview_speaks_the_typed_line(self):
        response, calls = self._preview(
            '/api/books/1/characters/7/preview', {'sample_text': 'Tűz és vér.'}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls['sample_text'], 'Tűz és vér.')

    def test_character_preview_without_a_line_keeps_the_named_default(self):
        _, calls = self._preview('/api/books/1/characters/7/preview', {})

        self.assertEqual(calls['sample_text'], app_module._character_preview_text('Aegon'))
        self.assertIn('Aegon', calls['sample_text'])

    def test_character_list_exposes_the_default_the_form_prefills(self):
        characters = self.client.get('/api/books/1/characters').get_json()

        self.assertEqual(
            characters[0]['preview_text'], app_module._character_preview_text('Aegon')
        )

    # ── limits and isolation ─────────────────────────────────────────────────

    def test_an_over_long_line_is_capped(self):
        """One preview is one synthesis call — on a cloud engine, a billed one."""
        _, calls = self._preview(
            '/api/books/1/characters/7/preview',
            {'sample_text': 'a' * (app_module.PREVIEW_TEXT_MAX_CHARS + 500)},
        )

        self.assertEqual(len(calls['sample_text']), app_module.PREVIEW_TEXT_MAX_CHARS)

    def test_a_non_string_line_is_ignored_rather_than_crashing(self):
        _, calls = self._preview(
            '/api/books/1/characters/7/preview', {'sample_text': {'not': 'a string'}}
        )

        self.assertEqual(calls['sample_text'], app_module._character_preview_text('Aegon'))

    def test_preview_text_is_never_persisted(self):
        self._preview(
            '/api/books/1/characters/7/preview', {'sample_text': 'Egyszeri próba.'}
        )
        self._preview(
            '/api/books/1/characters/narrator/preview', {'sample_text': 'Egyszeri próba.'}
        )

        with database.get_conn() as conn:
            character = dict(conn.execute(
                'SELECT * FROM characters WHERE id=7'
            ).fetchone())
            book = dict(conn.execute('SELECT * FROM books WHERE id=1').fetchone())
        self.assertNotIn('Egyszeri próba.', str(character.values()))
        self.assertNotIn('Egyszeri próba.', str(book.values()))

    def test_the_studio_page_prefills_the_narrator_default(self):
        page = self.client.get('/voice-studio/1').get_data(as_text=True)

        self.assertIn('narrator-preview-text', page)
        self.assertIn(app_module.VOICE_PREVIEW_TEXTS['hu'], page)


class PreviewDownloadTest(unittest.TestCase):
    """The Download button reuses the cached preview WAV as an attachment."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original_cache_dir = tts_engine.AUDIO_CACHE_DIR
        tts_engine.AUDIO_CACHE_DIR = self.tmp.name
        self.addCleanup(
            lambda: setattr(tts_engine, 'AUDIO_CACHE_DIR', self.original_cache_dir)
        )
        sf.write(
            os.path.join(self.tmp.name, 'abc123.wav'),
            np.zeros(2400, dtype=np.float32),
            24_000,
        )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()
        self.original_startup = app_module._startup_complete
        app_module._startup_complete = True
        self.addCleanup(
            lambda: setattr(app_module, '_startup_complete', self.original_startup)
        )

    def _get(self, url):
        response = self.client.get(url)
        self.addCleanup(response.close)
        return response

    def test_playback_still_streams_without_a_disposition(self):
        response = self._get('/api/audio/abc123')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'audio/wav')
        self.assertNotIn('attachment', response.headers.get('Content-Disposition', ''))

    def test_download_serves_the_same_wav_as_a_named_attachment(self):
        response = self._get('/api/audio/abc123?download=Aegon-preview')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'audio/wav')
        disposition = response.headers['Content-Disposition']
        self.assertIn('attachment', disposition)
        self.assertIn('Aegon-preview.wav', disposition)

    def test_an_accented_character_name_survives(self):
        """Werkzeug sends RFC 5987 filename* plus an ASCII fallback."""
        response = self._get('/api/audio/abc123?download=Sárkánykő-preview')

        disposition = response.headers['Content-Disposition']
        self.assertIn(
            "filename*=UTF-8''S%C3%A1rk%C3%A1nyk%C5%91-preview.wav", disposition
        )
        self.assertIn('filename=Sarkanyko-preview.wav', disposition)

    def test_a_missing_key_is_still_404(self):
        self.assertEqual(self._get('/api/audio/nope?download=x').status_code, 404)


class DownloadNameTest(unittest.TestCase):
    """Character names come from an LLM, so they reach this unvetted."""

    def test_path_separators_and_quotes_are_stripped(self):
        self.assertEqual(
            app_module._download_name('../../etc/passwd'), 'etcpasswd.wav'
        )
        self.assertEqual(app_module._download_name('a"b;c'), 'abc.wav')

    def test_accents_spaces_and_hyphens_are_kept(self):
        self.assertEqual(
            app_module._download_name('Sárkánykő úr-preview'),
            'Sárkánykő úr-preview.wav',
        )

    def test_an_empty_or_symbol_only_name_falls_back(self):
        self.assertEqual(app_module._download_name(''), 'preview.wav')
        self.assertEqual(app_module._download_name('***'), 'preview.wav')
        self.assertEqual(app_module._download_name(None), 'preview.wav')

    def test_a_very_long_name_is_trimmed(self):
        name = app_module._download_name('x' * 500)

        self.assertEqual(name, 'x' * 60 + '.wav')


if __name__ == '__main__':
    unittest.main()
