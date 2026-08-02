"""Browser tests for the TTS engine selector on the settings page.

The engine panels are shown and hidden entirely in the browser, and every
field is read back by one loadSettings() pass — a single missing element id
throws there and silently leaves the rest of the page unconfigured. That only
shows up in a real DOM.

Requires Playwright and its browser; see tests/test_ui_pronunciation.py.
"""

import logging
import tempfile
import threading
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    sync_playwright = None

import app as app_module
from core import database, settings


def _chromium_available():
    if sync_playwright is None:
        return False
    try:
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_chromium_available(), 'Playwright chromium is not available')
class EngineSettingsUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from werkzeug.serving import make_server

        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls._db_path = database.DB_PATH
        cls._settings_file = settings.SETTINGS_FILE
        cls._startup = app_module._startup_complete

        database.DB_PATH = str(tmp / 'reader.db')
        settings.SETTINGS_FILE = tmp / 'settings.json'
        app_module._startup_complete = True
        database.init_db()

        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        cls._server = make_server('127.0.0.1', 0, app_module.app)
        cls.base = f'http://127.0.0.1:{cls._server.server_port}'
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

        # /api/tts/status auto-loads whichever engine is selected, and the page
        # polls it on every load and after every save. Left alone, saving Higgs
        # or Piper starts a real multi-gigabyte download inside this process:
        # it blocks the single-threaded test server, times out the tests that
        # follow, and then segfaults when teardown races the loading thread.
        # These tests are about the form, never about loading a model.
        cls._load_async = app_module.tts.load_async
        app_module.tts.load_async = lambda: None

        cls._playwright = sync_playwright().start()
        cls._browser = cls._playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        app_module.tts.load_async = cls._load_async
        cls._browser.close()
        cls._playwright.stop()
        cls._server.shutdown()
        cls._thread.join(timeout=5)
        database.DB_PATH = cls._db_path
        settings.SETTINGS_FILE = cls._settings_file
        app_module._startup_complete = cls._startup
        cls._tmp.cleanup()

    def setUp(self):
        # Every page polls /api/tts/status, which loads the selected engine.
        # Staying on ElevenLabs keeps that a no-op, and pointing it at this
        # test server means no request ever leaves the machine.
        settings.save({
            'tts_engine': 'elevenlabs',
            'elevenlabs_base_url': self.base,
            'elevenlabs_api_key': '',
            'elevenlabs_voice_id': '',
            'higgs_quantization': '4bit',
            'higgs_concurrency': 2,
        })
        self.page = self._browser.new_page()
        self.errors = []
        self.page.on('pageerror', lambda e: self.errors.append(str(e)))
        self.addCleanup(self.page.close)
        self.addCleanup(lambda: self.assertEqual(self.errors, []))

    def _open_settings(self):
        self.page.goto(f'{self.base}/settings', wait_until='domcontentloaded')
        # loadSettings() is async and sets this flag last. Waiting on a field
        # value instead would race: every select already has its first option
        # selected before the fetch resolves, so the wait would pass early and
        # the test would read HTML defaults. The flag is a top-level `let`, so
        # it lives in the global lexical scope and is not a window property —
        # it has to be referenced by bare name.
        self.page.wait_for_function('_settingsReady === true')
        return self.page

    def test_engine_selector_swaps_the_visible_engine_panel(self):
        page = self._open_settings()

        self.assertTrue(page.locator('.elevenlabs-settings').first.is_visible())
        self.assertFalse(page.locator('.omnivoice-settings').first.is_visible())
        self.assertFalse(page.locator('.higgs-settings').first.is_visible())

        page.select_option('#tts-engine', 'omnivoice')

        self.assertTrue(page.locator('.omnivoice-settings').first.is_visible())
        self.assertFalse(page.locator('.elevenlabs-settings').first.is_visible())

        page.select_option('#tts-engine', 'f5')

        self.assertTrue(page.locator('.f5-settings').first.is_visible())
        self.assertFalse(page.locator('.omnivoice-settings').first.is_visible())

        page.select_option('#tts-engine', 'piper')

        self.assertTrue(page.locator('.piper-settings').first.is_visible())
        self.assertFalse(page.locator('.f5-settings').first.is_visible())

        page.select_option('#tts-engine', 'elevenlabs')

        self.assertTrue(page.locator('.elevenlabs-settings').first.is_visible())
        self.assertFalse(page.locator('.omnivoice-settings').first.is_visible())
        self.assertFalse(page.locator('.f5-settings').first.is_visible())
        self.assertFalse(page.locator('.piper-settings').first.is_visible())

    def test_higgs_quantization_round_trips_through_save_and_reload(self):
        """Saving from a half-loaded form would silently reset this to default."""
        page = self._open_settings()
        page.select_option('#tts-engine', 'higgs')
        self.assertEqual(page.input_value('#higgs-quantization'), '4bit')

        page.select_option('#higgs-quantization', '8bit')
        page.click('.settings-footer .btn-primary')
        page.wait_for_function(
            "document.getElementById('save-hint').textContent.length > 0"
        )

        saved = settings.load()
        self.assertEqual(saved['higgs_quantization'], '8bit')

    def test_saving_does_not_reset_the_hidden_concurrency_knob(self):
        """It has no control on the page, so the form must leave it alone."""
        page = self._open_settings()
        page.select_option('#tts-engine', 'higgs')
        page.click('.settings-footer .btn-primary')
        page.wait_for_function(
            "document.getElementById('save-hint').textContent.length > 0"
        )

        self.assertEqual(settings.load()['higgs_concurrency'], 2)

    def test_elevenlabs_fields_round_trip_through_save_and_reload(self):
        page = self._open_settings()
        page.fill('#elevenlabs-api-key', 'sk_ui_test')
        page.fill('#elevenlabs-voice-id', '  voice-42  ')
        page.select_option('#elevenlabs-model-id', 'eleven_turbo_v2_5')
        page.select_option('#elevenlabs-output-format', 'pcm_24000')
        page.fill('#elevenlabs-stability', '0.35')
        page.uncheck('#elevenlabs-speaker-boost')
        page.click('.settings-footer .btn-primary')
        page.wait_for_function(
            "document.getElementById('save-hint').textContent.length > 0"
        )

        saved = settings.load()
        self.assertEqual(saved['tts_engine'], 'elevenlabs')
        self.assertEqual(saved['elevenlabs_voice_id'], 'voice-42')
        self.assertEqual(saved['elevenlabs_model_id'], 'eleven_turbo_v2_5')
        self.assertEqual(saved['elevenlabs_output_format'], 'pcm_24000')
        self.assertAlmostEqual(saved['elevenlabs_stability'], 0.35)
        self.assertIs(saved['elevenlabs_speaker_boost'], False)

        page = self._open_settings()
        self.assertTrue(page.locator('.elevenlabs-settings').first.is_visible())
        self.assertEqual(page.input_value('#elevenlabs-voice-id'), 'voice-42')
        self.assertEqual(page.input_value('#elevenlabs-stability'), '0.35')
        self.assertFalse(page.is_checked('#elevenlabs-speaker-boost'))

    def test_f5_fields_round_trip_through_save_and_reload(self):
        # Every f5_* key needs its own entry in the /api/settings allow-list;
        # a missing one is dropped server-side with no error anywhere.
        page = self._open_settings()
        page.select_option('#tts-engine', 'f5')
        page.fill('#f5-ref-audio', '  /tmp/narrator.wav  ')
        page.fill('#f5-ref-text', 'Jó napot kívánok.')
        page.fill('#f5-nfe-step', '24')
        page.fill('#f5-cfg-strength', '1.8')
        page.uncheck('#f5-trim-onset')
        page.click('.settings-footer .btn-primary')
        page.wait_for_function(
            "document.getElementById('save-hint').textContent.length > 0"
        )

        saved = settings.load()
        self.assertEqual(saved['tts_engine'], 'f5')
        self.assertEqual(saved['f5_ref_audio'], '/tmp/narrator.wav')
        self.assertEqual(saved['f5_ref_text'], 'Jó napot kívánok.')
        self.assertEqual(saved['f5_nfe_step'], 24)
        self.assertAlmostEqual(saved['f5_cfg_strength'], 1.8)
        self.assertIs(saved['f5_trim_onset'], False)

        page = self._open_settings()
        self.assertTrue(page.locator('.f5-settings').first.is_visible())
        self.assertEqual(page.input_value('#f5-ref-audio'), '/tmp/narrator.wav')
        self.assertEqual(page.input_value('#f5-nfe-step'), '24')
        self.assertFalse(page.is_checked('#f5-trim-onset'))

    def _save_piper_fields_without_activating_it(self, page):
        """Persist the Piper form while leaving ElevenLabs as the active engine.

        saveSettings() reads the DOM regardless of which panel is visible, so
        switching the selector back before saving still stores every piper_*
        field. Actually persisting 'piper' would make the next page load poll
        /api/tts/status, really load the engine, and fetch ~60 MB of voices in a
        daemon thread that then races the class teardown — a hard crash, not a
        test failure.
        """
        page.select_option('#tts-engine', 'elevenlabs')
        page.click('.settings-footer .btn-primary')
        page.wait_for_function(
            "document.getElementById('save-hint').textContent.length > 0"
        )

    def test_piper_fields_round_trip_through_save_and_reload(self):
        page = self._open_settings()
        page.select_option('#tts-engine', 'piper')
        page.fill('#piper-narrator-voice', '  hu_HU-imre-medium  ')
        page.fill('#piper-character-voices', 'hu_HU-anna-medium, hu_HU-berta-medium')
        page.fill('#piper-length-scale', '1.15')
        page.uncheck('#piper-match-gender')
        self._save_piper_fields_without_activating_it(page)

        saved = settings.load()
        self.assertEqual(saved['piper_narrator_voice'], 'hu_HU-imre-medium')
        self.assertEqual(
            saved['piper_character_voices'], 'hu_HU-anna-medium,hu_HU-berta-medium'
        )
        self.assertAlmostEqual(saved['piper_length_scale'], 1.15)
        self.assertIs(saved['piper_match_gender'], False)

        page = self._open_settings()
        page.select_option('#tts-engine', 'piper')
        self.assertTrue(page.locator('.piper-settings').first.is_visible())
        self.assertEqual(page.input_value('#piper-narrator-voice'), 'hu_HU-imre-medium')
        self.assertEqual(page.input_value('#piper-length-scale'), '1.15')
        self.assertFalse(page.is_checked('#piper-match-gender'))

    def test_a_malformed_piper_voice_name_is_rejected_on_save(self):
        """It would otherwise fail as a download 404 in the middle of a chapter."""
        page = self._open_settings()
        page.select_option('#tts-engine', 'piper')
        page.fill('#piper-narrator-voice', 'not-a-voice')
        page.fill('#piper-character-voices', 'hu_HU-imre-medium,garbage,,')
        self._save_piper_fields_without_activating_it(page)

        saved = settings.load()
        self.assertEqual(saved['piper_narrator_voice'], 'hu_HU-anna-medium')
        self.assertEqual(saved['piper_character_voices'], 'hu_HU-imre-medium')

    def test_piper_voice_source_swaps_its_sub_panels(self):
        page = self._open_settings()
        page.select_option('#tts-engine', 'piper')
        # Set the starting state explicitly: an earlier test may have persisted
        # the local source.
        page.check('#piper-src-download')

        self.assertTrue(page.locator('#piper-panel-download').is_visible())
        self.assertFalse(page.locator('#piper-panel-local').is_visible())

        page.check('#piper-src-local')

        self.assertTrue(page.locator('#piper-panel-local').is_visible())
        self.assertFalse(page.locator('#piper-panel-download').is_visible())

    def test_f5_model_source_swaps_its_sub_panels(self):
        page = self._open_settings()
        page.select_option('#tts-engine', 'f5')

        self.assertTrue(page.locator('#f5-panel-download').is_visible())
        self.assertFalse(page.locator('#f5-panel-local').is_visible())

        page.check('#f5-src-local')

        self.assertTrue(page.locator('#f5-panel-local').is_visible())
        self.assertFalse(page.locator('#f5-panel-download').is_visible())

    def test_unconfigured_engine_reports_what_is_missing(self):
        page = self._open_settings()

        page.click("button:has-text('Reload TTS engine')")
        hint = page.locator('#tts-reload-hint')
        page.wait_for_function(
            "document.getElementById('tts-reload-hint').textContent.includes('failed')",
            timeout=15000,
        )
        self.assertIn('API key', hint.inner_text())
        self.assertIn('API key', page.locator('#elevenlabs-status').inner_text())


if __name__ == '__main__':
    unittest.main()
