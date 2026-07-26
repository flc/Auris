import unittest

from core.enrichment import MAX_TTS_SEGMENT_CHARS, enrich_chapter


class EnrichmentTextPreparationTests(unittest.TestCase):
    def test_import_artifacts_are_removed_before_synthesis(self):
        segments = enrich_chapter(
            "\ufeffElső\u00a0mondat.\u200b\x00 Második mondat.",
            character_map={},
        )

        spoken = " ".join(segment["text"] for segment in segments)
        self.assertEqual(spoken, "Első mondat. Második mondat.")
        self.assertNotIn("\ufeff", spoken)
        self.assertNotIn("\u200b", spoken)

    def test_pathological_long_sentence_is_split_safely(self):
        long_sentence = " ".join(["hosszúszó"] * 140) + "."

        segments = enrich_chapter(long_sentence, character_map={})

        self.assertGreater(len(segments), 1)
        self.assertTrue(
            all(len(segment["text"]) <= MAX_TTS_SEGMENT_CHARS for segment in segments)
        )
        self.assertTrue(segments[-1]["ends_paragraph"])
        self.assertTrue(
            all(not segment["ends_paragraph"] for segment in segments[:-1])
        )

    def test_only_last_segment_of_each_paragraph_marks_paragraph_end(self):
        segments = enrich_chapter(
            "Első mondat. Második mondat.\n\nHarmadik mondat.",
            character_map={},
        )

        self.assertEqual(
            [segment["ends_paragraph"] for segment in segments],
            [True, True],
        )


if __name__ == "__main__":
    unittest.main()
