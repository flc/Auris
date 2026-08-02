import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import exporter


class ChapterSelectionTests(unittest.TestCase):
    def test_all_selects_every_chapter(self):
        for value in (None, '', '*', 'all', 'mind', 'összes'):
            with self.subTest(value=value):
                self.assertEqual(exporter.parse_chapter_selection(value, 4), [1, 2, 3, 4])

    def test_numbers_ranges_and_duplicates_are_sorted(self):
        self.assertEqual(
            exporter.parse_chapter_selection('5, 1, 3-4, 3', 6),
            [1, 3, 4, 5],
        )

    def test_invalid_selection_is_rejected(self):
        for value in ('0', '1,,2', '4-2', '1,a', '7'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    exporter.parse_chapter_selection(value, 6)


class SegmentSelectionTests(unittest.TestCase):
    def test_all_selects_every_segment(self):
        for value in (None, '', '*', 'all', 'mind', 'összes'):
            with self.subTest(value=value):
                self.assertEqual(exporter.parse_segment_selection(value, 3), [1, 2, 3])

    def test_numbers_ranges_and_duplicates_are_sorted(self):
        self.assertEqual(
            exporter.parse_segment_selection('9, 2, 4-6, 4', 12),
            [2, 4, 5, 6, 9],
        )

    def test_invalid_selection_is_rejected(self):
        for value in ('0', '1,,2', '6-2', '2,x', '13'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    exporter.parse_segment_selection(value, 12)

    def test_error_text_names_segments_not_chapters(self):
        with self.assertRaises(ValueError) as caught:
            exporter.parse_segment_selection('40', 12)
        self.assertIn('Segment number out of range', str(caught.exception))

    def test_empty_chapter_reports_segments(self):
        with self.assertRaises(ValueError) as caught:
            exporter.parse_segment_selection('all', 0)
        self.assertIn('no segments', str(caught.exception))


class SelectionFormattingTests(unittest.TestCase):
    def test_consecutive_numbers_collapse_into_ranges(self):
        self.assertEqual(exporter.format_selection([1, 2, 3, 7]), '1-3_7')

    def test_scattered_numbers_stay_separate(self):
        self.assertEqual(exporter.format_selection([4, 9, 11]), '4_9_11')

    def test_single_number(self):
        self.assertEqual(exporter.format_selection([5]), '5')

    def test_result_survives_filename_sanitizing(self):
        # _safe_name drops commas, so "1,3" would collapse to the number 13.
        stem = exporter._safe_name(f'Chapter 1_seg_{exporter.format_selection([1, 3])}')
        self.assertEqual(stem, 'Chapter_1_seg_1_3')


class ChapterFolderExportTests(unittest.TestCase):
    def test_mastering_falls_back_to_unprocessed_wav_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.wav')
            sf.write(
                source,
                np.sin(np.linspace(0, np.pi * 20, 2400)).astype(np.float32),
                exporter.SAMPLE_RATE,
            )
            with (
                patch.object(exporter, 'EXPORTS_DIR', tmp),
                patch.object(exporter, '_ffmpeg_available', return_value=False),
            ):
                result = exporter.export_single_chapter(
                    'Chapter',
                    'Book',
                    [{
                        'audio_path': source,
                        'duration_sec': 0.1,
                        'text': 'Text',
                    }],
                    {},
                    mastering=True,
                    book_author='Writer',
                )

            self.assertTrue(os.path.isfile(result['audio_path']))
            self.assertEqual(
                os.path.dirname(result['audio_path']),
                os.path.join(tmp, 'Writer - Book'),
            )
            self.assertFalse(result['mastering_applied'])
            self.assertIn('FFmpeg', result['mastering_warning'])
            self.assertFalse(
                os.path.exists(
                    os.path.join(tmp, 'Writer - Book', '.Chapter.premaster.wav')
                )
            )

    def test_loudnorm_measurements_are_parsed_from_ffmpeg_output(self):
        measurements = exporter._extract_loudnorm_measurements(
            'noise before\n'
            '{\n'
            '  "input_i" : "-22.10",\n'
            '  "input_tp" : "-4.20",\n'
            '  "input_lra" : "3.50",\n'
            '  "input_thresh" : "-32.20",\n'
            '  "output_i" : "-19.00",\n'
            '  "target_offset" : "0.10"\n'
            '}\n'
        )

        self.assertEqual(measurements['input_i'], '-22.10')
        self.assertEqual(measurements['target_offset'], '0.10')

    def test_export_creates_book_folder_with_numbered_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.wav')
            sf.write(source, np.zeros(100, dtype=np.float32), exporter.SAMPLE_RATE)
            chapters = [
                {
                    'chapter_number': 2,
                    'chapter_title': 'The Beginning',
                    'segments': [{
                        'audio_path': source,
                        'duration_sec': 100 / exporter.SAMPLE_RATE,
                        'text': 'Hello.',
                    }],
                },
                {
                    'chapter_number': 11,
                    'chapter_title': 'The End',
                    'segments': [{
                        'audio_path': source,
                        'duration_sec': 100 / exporter.SAMPLE_RATE,
                        'text': 'Goodbye.',
                    }],
                },
            ]

            with patch.object(exporter, 'EXPORTS_DIR', tmp):
                result = exporter.export_chapter_folder(
                    'My Book', chapters, {}, audio_fmt='wav', sub_fmt='srt',
                    book_author='Jane Writer',
                )

            self.assertEqual(
                result['directory_path'],
                os.path.join(tmp, 'Jane Writer - My Book'),
            )
            self.assertTrue(os.path.isfile(os.path.join(result['directory_path'], '02_The_Beginning.wav')))
            self.assertTrue(os.path.isfile(os.path.join(result['directory_path'], '02_The_Beginning.srt')))
            self.assertTrue(os.path.isfile(os.path.join(result['directory_path'], '11_The_End.wav')))

    def test_successful_mp3_conversion_removes_intermediate_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.wav')
            sf.write(source, np.zeros(100, dtype=np.float32), exporter.SAMPLE_RATE)
            with (
                patch.object(exporter, 'EXPORTS_DIR', tmp),
                patch.object(
                    exporter,
                    '_wav_to_mp3_bytes',
                    return_value=b'mp3',
                ) as encode_mp3,
            ):
                result = exporter.export_single_chapter(
                    'Chapter', 'Book',
                    [{'audio_path': source, 'duration_sec': 0.1, 'text': 'Text'}],
                    {}, audio_fmt='mp3', sub_fmt='srt',
                    book_author='Writer',
                )

            self.assertTrue(os.path.isfile(result['audio_path']))
            encode_mp3.assert_called_once()
            self.assertEqual(
                encode_mp3.call_args.kwargs['tags'],
                {
                    'title': 'Chapter',
                    'artist': 'Writer',
                    'album': 'Book',
                    'track': '',
                },
            )
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 'Writer - Book', 'Chapter.wav'))
            )

    def test_book_folder_removes_invalid_filename_characters(self):
        with patch.object(exporter, 'EXPORTS_DIR', 'exports'):
            path = exporter._book_export_dir('A: Writer', 'Book? <One>')

        self.assertEqual(path, os.path.join('exports', 'A Writer - Book One'))


if __name__ == '__main__':
    unittest.main()
