"""Detecting sentences whose audio is too short for their text.

The point of the detector is that it needs no listener and no hardcoded speech
rate: a chapter tells you its own pace, and a sentence far above it lost words.
These pin the two ways that can go wrong — missing a genuine truncation, and
flagging a sentence that is merely brisk.
"""

import unittest

from core import audio_check


def _segment(text, duration, audio=True):
    return {
        'text': text,
        'duration_sec': duration,
        'audio_path': '/tmp/x.wav' if audio else None,
    }


# 100 characters read at 15 chars/sec is a normal 6.7 second sentence.
NORMAL = 'x' * 100


def _chapter(count=10, duration=6.7):
    return [_segment(NORMAL, duration) for _ in range(count)]


class SpeechRateTests(unittest.TestCase):
    def test_rate_is_characters_over_seconds(self):
        self.assertAlmostEqual(audio_check.speech_rate('x' * 90, 6.0), 15.0)

    def test_short_lines_carry_no_signal(self):
        """'Igen.' is mostly silence; its rate says nothing."""
        self.assertIsNone(audio_check.speech_rate('Igen.', 1.2))

    def test_missing_duration_is_not_a_rate(self):
        self.assertIsNone(audio_check.speech_rate(NORMAL, 0))
        self.assertIsNone(audio_check.speech_rate(NORMAL, None))


class MedianRateTests(unittest.TestCase):
    def test_the_chapter_sets_its_own_pace(self):
        self.assertAlmostEqual(audio_check.median_speech_rate(_chapter()), 100 / 6.7)

    def test_a_thin_chapter_refuses_to_guess(self):
        self.assertIsNone(audio_check.median_speech_rate(_chapter(count=3)))

    def test_short_lines_do_not_pollute_the_median(self):
        segments = _chapter() + [_segment('Igen.', 0.1) for _ in range(20)]

        self.assertAlmostEqual(audio_check.median_speech_rate(segments), 100 / 6.7)


class SuspectTests(unittest.TestCase):
    def test_a_truncated_sentence_is_flagged(self):
        segments = _chapter()
        segments.append(_segment(NORMAL, 2.0))  # 50 chars/sec — words are gone

        suspects = audio_check.find_suspects(segments)

        self.assertEqual([s['index'] for s in suspects], [len(segments) - 1])
        self.assertIn('missing', suspects[0]['reason'])

    def test_a_merely_brisk_sentence_is_left_alone(self):
        segments = _chapter()
        segments.append(_segment(NORMAL, 5.0))  # 1.34x the median

        self.assertEqual(audio_check.find_suspects(segments), [])

    def test_a_slow_sentence_is_never_suspect(self):
        """Long audio for short text is a pause, not a loss."""
        segments = _chapter()
        segments.append(_segment(NORMAL, 20.0))

        self.assertEqual(audio_check.find_suspects(segments), [])

    def test_silence_for_real_text_is_flagged_without_a_median(self):
        segments = [_segment(NORMAL, 0.05)]

        suspects = audio_check.find_suspects(segments)

        self.assertEqual(len(suspects), 1)
        self.assertIn('no audio', suspects[0]['reason'])

    def test_nothing_is_flagged_on_rate_alone_without_enough_reference(self):
        segments = _chapter(count=3) + [_segment(NORMAL, 2.0)]

        self.assertEqual(audio_check.find_suspects(segments), [])

    def test_segments_without_audio_are_not_the_detector_s_problem(self):
        segments = _chapter() + [_segment(NORMAL, None, audio=False)]

        self.assertEqual(audio_check.find_suspects(segments), [])

    def test_a_short_line_is_never_flagged_on_rate(self):
        segments = _chapter()
        segments.append(_segment('Igen.', 0.4))

        self.assertEqual(audio_check.find_suspects(segments), [])


class AcceptanceTests(unittest.TestCase):
    def test_a_retake_back_in_range_is_accepted(self):
        median = 100 / 6.7

        self.assertTrue(audio_check.is_acceptable(NORMAL, 6.5, median))
        self.assertFalse(audio_check.is_acceptable(NORMAL, 2.0, median))

    def test_silence_is_never_acceptable(self):
        self.assertFalse(audio_check.is_acceptable(NORMAL, 0.05, 15.0))
        self.assertFalse(audio_check.is_acceptable(NORMAL, None, 15.0))

    def test_without_a_median_anything_audible_passes(self):
        self.assertTrue(audio_check.is_acceptable(NORMAL, 1.0, None))


if __name__ == '__main__':
    unittest.main()
