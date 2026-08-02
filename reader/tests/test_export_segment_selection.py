"""Segment selection on the single-chapter export.

The scope is one chapter, so the numbers are 1-based within it — the same
numbering the playback counter shows. A partial export must synthesize only the
chosen segments and must not overwrite the whole-chapter file.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

import app as app_module
from core import database


class _FakeTTS:
    engine_name = "piper"

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir

    def status(self):
        return {"state": "ready"}

    def load_async(self):
        return None

    def generate_many(self, items, num_step=None, batch_size=None,
                      on_item=None, on_status=None):
        results = []
        for index, _item in enumerate(items):
            path = os.path.join(self.audio_dir, f"generated-{index}.wav")
            with open(path, "wb") as audio_file:
                audio_file.write(b"RIFF-test")
            result = {
                "audio_path": path,
                "duration_sec": 1.0,
                "cache_hit": False,
                "cache_key": f"generated-{index}",
            }
            results.append(result)
            if on_item:
                on_item(index, result)
        return results

    def set_dedicated_cuda_stream(self, enabled):
        return None


class ExportSegmentSelectionTest(unittest.TestCase):
    SEGMENT_TEXTS = ("One.", "Two.", "Three.", "Four.", "Five.")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        self.original_startup = app_module._startup_complete
        self.original_tts = app_module.tts
        database.DB_PATH = os.path.join(self.tmp.name, "reader.db")
        app_module._startup_complete = True
        app_module.tts = _FakeTTS(self.tmp.name)
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, author, file_path, file_type, language) "
                "VALUES (1, 'Test', 'Author', 'test.txt', 'txt', 'en')"
            )
            conn.execute(
                "INSERT INTO chapters "
                "(id, book_id, title, order_num, content, word_count) "
                "VALUES (2, 1, 'Chapter 1', 0, 'One. Two. Three. Four. Five.', 5)"
            )
            for index, text in enumerate(self.SEGMENT_TEXTS):
                conn.execute(
                    "INSERT INTO tts_segments "
                    "(book_id, chapter_id, segment_index, text, enriched_text, "
                    "instruct, speed, is_dialogue, cache_key) "
                    "VALUES (1, 2, ?, ?, ?, 'narrator', 1.0, 0, ?)",
                    (index, text, text, f"pending-{index}"),
                )
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        app_module._startup_complete = self.original_startup
        app_module.tts = self.original_tts
        self.tmp.cleanup()

    def _export(self, payload):
        """POST an export and return (response, captured export_single_chapter call)."""
        captured = {}

        def fake_export(chapter_title, book_title, segments, colors, audio_fmt,
                        sub_fmt, file_stem=None, **kwargs):
            captured["texts"] = [seg["text"] for seg in segments]
            captured["file_stem"] = file_stem
            return {
                "audio_path": os.path.join(self.tmp.name, "out.wav"),
                "subtitle_path": os.path.join(self.tmp.name, "out.srt"),
                "mastering_applied": False,
            }

        with (
            patch.object(app_module.exporter, "export_single_chapter", fake_export),
            patch.object(app_module, "_start_export_pool", return_value=None),
        ):
            response = self.client.post(
                "/api/books/1/export/chapter/2", json=payload
            )
            if response.status_code == 200:
                job_id = response.get_json()["job_id"]
                for _ in range(200):
                    job = self.client.get(
                        f"/api/export/status/{job_id}"
                    ).get_json()
                    if job["state"] in ("complete", "failed"):
                        break
                    time.sleep(0.01)
                captured["job"] = job
        return response, captured

    def test_the_reader_offers_the_field_and_defaults_it_to_all(self):
        page = self.client.get("/reader/1")

        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="exp-segments"', page.data)
        self.assertIn(b'id="segment-selection-wrap"', page.data)
        # Visible with the default "This chapter" scope, unlike the chapter box.
        self.assertNotIn(
            b'id="segment-selection-wrap" class="chapter-selection hidden"', page.data
        )

    def test_selecting_a_range_exports_only_those_segments(self):
        response, captured = self._export(
            {"audio_fmt": "wav", "sub_fmt": "srt", "segments": "2-3"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["job"]["state"], "complete")
        self.assertEqual(captured["texts"], ["Two.", "Three."])

    def test_scattered_numbers_are_exported_in_reading_order(self):
        _, captured = self._export({"segments": "5, 1, 3"})

        self.assertEqual(captured["texts"], ["One.", "Three.", "Five."])

    def test_partial_export_does_not_overwrite_the_whole_chapter_file(self):
        _, captured = self._export({"segments": "2-3"})

        self.assertEqual(captured["file_stem"], "Chapter 1_seg_2-3")

    def test_all_keeps_the_plain_chapter_filename(self):
        for value in ("all", "", None):
            with self.subTest(value=value):
                _, captured = self._export({"segments": value})
                self.assertIsNone(captured["file_stem"])
                self.assertEqual(len(captured["texts"]), len(self.SEGMENT_TEXTS))

    def test_a_missing_segments_field_exports_the_whole_chapter(self):
        _, captured = self._export({"audio_fmt": "wav"})

        self.assertIsNone(captured["file_stem"])
        self.assertEqual(len(captured["texts"]), len(self.SEGMENT_TEXTS))

    def test_out_of_range_number_is_rejected_before_the_job_starts(self):
        response, captured = self._export({"segments": "9"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Segment number out of range", response.get_json()["error"])
        self.assertNotIn("texts", captured)

    def test_malformed_selection_is_rejected(self):
        response, _ = self._export({"segments": "2,x"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("segment numbers", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
