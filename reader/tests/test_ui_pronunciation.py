"""Browser tests for the pronunciation dictionary editor.

The editor is client-side, so the behaviour that matters — filtering, adding a
rule, keeping focus while typing, flagging duplicates — only exists in a real
DOM. These run against a throwaway app instance on a random port.

Requires Playwright and its browser:

    .venv/bin/pip install playwright
    .venv/bin/playwright install chromium
    sudo .venv/bin/playwright install-deps chromium   # Linux/WSL system libs

They skip when either is missing, so the normal suite stays dependency-free.
"""

import logging
import sqlite3
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

GLOBAL_RULES = '\n'.join([
    '# Westerosi names',
    'Westeros = Veszterosz',
    'Aegon = Egon',
    'Targaryen = Targerjen',
])
BOOK_RULES = 'Aegon = Egon\nSárkánykő = Sárkánykő'


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
class PronunciationEditorUITest(unittest.TestCase):
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
        settings.save({'pronunciation_dict': GLOBAL_RULES})
        conn = sqlite3.connect(database.DB_PATH)
        conn.execute(
            "INSERT INTO books (id, title, file_path, file_type, language, "
            "narrator_instruct, pronunciation_dict) "
            "VALUES (1, 'Test', 't.txt', 'txt', 'hu', 'male, elderly', ?)",
            (BOOK_RULES,),
        )
        conn.commit()
        conn.close()

        # Every page load logs a dozen requests; they bury the test results.
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        cls._server = make_server('127.0.0.1', 0, app_module.app)
        cls.base = f'http://127.0.0.1:{cls._server.server_port}'
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

        cls._playwright = sync_playwright().start()
        cls._browser = cls._playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls._browser.close()
        cls._playwright.stop()
        cls._server.shutdown()
        cls._thread.join(timeout=5)
        database.DB_PATH = cls._db_path
        settings.SETTINGS_FILE = cls._settings_file
        app_module._startup_complete = cls._startup
        cls._tmp.cleanup()

    def setUp(self):
        self.page = self._browser.new_page()
        self.errors = []
        self.page.on('pageerror', lambda e: self.errors.append(str(e)))
        self.addCleanup(self.page.close)
        self.addCleanup(lambda: self.assertEqual(self.errors, []))

    def _open_settings(self):
        # The dictionary lives in the "Generation & export" category, and the
        # page reads the category from the hash on load.
        self.page.goto(f'{self.base}/settings#output', wait_until='domcontentloaded')
        self.page.locator('#pronunciation-dict .lexicon-table').wait_for()
        return self.page.locator('#pronunciation-dict')

    def test_global_rules_are_listed_as_rows(self):
        editor = self._open_settings()

        self.assertEqual(editor.locator('tbody tr').count(), 3)
        self.assertEqual(
            editor.locator('[data-field="source"]').first.input_value(), 'Westeros')
        self.assertEqual(
            editor.locator('[data-field="spoken"]').first.input_value(), 'Veszterosz')
        self.assertIn('3', editor.locator('.lexicon-count').inner_text())

    def test_filter_narrows_the_list_without_touching_the_rules(self):
        editor = self._open_settings()

        editor.locator('.lexicon-filter-input').fill('targ')
        self.assertEqual(editor.locator('tbody tr').count(), 1)
        self.assertEqual(
            editor.locator('[data-field="source"]').first.input_value(), 'Targaryen')

        editor.locator('.lexicon-filter-input').fill('')
        self.assertEqual(editor.locator('tbody tr').count(), 3)

    def test_filtering_is_not_an_unsaved_change(self):
        editor = self._open_settings()
        banner = self.page.locator('#settings-unsaved-banner')

        editor.locator('.lexicon-filter-input').fill('aeg')
        self.assertFalse(banner.is_visible(), 'a filter only changes the view')

        editor.locator('[data-field="spoken"]').first.fill('Vesztirosz')
        self.assertTrue(banner.is_visible(), 'editing a rule needs saving')

    def test_added_rule_keeps_focus_while_both_cells_are_typed(self):
        editor = self._open_settings()

        editor.locator('.lexicon-add').click()
        self.page.keyboard.type('Rhaenys')
        self.page.keyboard.press('Enter')          # written -> spoken
        self.page.keyboard.type('Renisz')

        row = editor.locator('tbody tr').last
        self.assertEqual(row.locator('[data-field="source"]').input_value(), 'Rhaenys')
        self.assertEqual(row.locator('[data-field="spoken"]').input_value(), 'Renisz')

    def test_pasting_several_rules_becomes_several_rows(self):
        editor = self._open_settings()

        editor.locator('.lexicon-add').click()
        # A single-line input would flatten this into one cell, making the rest
        # of the block part of Aenar's pronunciation.
        self.page.evaluate(
            """() => {
                const cell = document.querySelector('#pronunciation-dict tbody tr:last-child [data-field="source"]');
                const data = new DataTransfer();
                data.setData('text/plain', 'Aenar = Enár\\nVelaryon = Velarion');
                cell.focus();
                cell.dispatchEvent(new ClipboardEvent('paste', {
                    clipboardData: data, bubbles: true, cancelable: true,
                }));
            }"""
        )

        rows = editor.locator('tbody tr')
        self.assertEqual(rows.count(), 5)
        self.assertEqual(rows.nth(3).locator('[data-field="source"]').input_value(), 'Aenar')
        self.assertEqual(rows.nth(3).locator('[data-field="spoken"]').input_value(), 'Enár')
        self.assertEqual(rows.nth(4).locator('[data-field="source"]').input_value(), 'Velaryon')
        self.assertEqual(rows.nth(4).locator('[data-field="spoken"]').input_value(), 'Velarion')

    def test_sorting_orders_the_rules_by_written_form(self):
        editor = self._open_settings()

        editor.locator('.lexicon-sort').click()

        written = editor.locator('[data-field="source"]').all()
        self.assertEqual(
            [cell.input_value() for cell in written],
            ['Aegon', 'Targaryen', 'Westeros'],
        )

    def test_repeated_written_form_is_flagged(self):
        editor = self._open_settings()

        editor.locator('.lexicon-add').click()
        self.page.keyboard.type('westeros')

        flagged = editor.locator('tr.lexicon-duplicate')
        self.assertEqual(flagged.count(), 1)
        self.assertIn('already exists', flagged.first.get_attribute('title'))

    def test_removing_a_rule_drops_the_row(self):
        editor = self._open_settings()

        editor.locator('tbody tr').first.locator('.lexicon-remove').click()

        self.assertEqual(editor.locator('tbody tr').count(), 2)
        self.assertEqual(
            editor.locator('[data-field="source"]').first.input_value(), 'Aegon')

    def test_book_editor_shows_only_the_book_rules(self):
        self.page.goto(f'{self.base}/voice-studio/1', wait_until='domcontentloaded')
        editor = self.page.locator('#book-pronunciation-dict')
        editor.locator('.lexicon-table').wait_for()

        self.assertEqual(editor.locator('tbody tr').count(), 2)
        self.assertEqual(
            editor.locator('[data-field="source"]').first.input_value(), 'Aegon')


if __name__ == '__main__':
    unittest.main()
