import re
import unittest

from core.pronunciation import (
    apply_pronunciation,
    entries,
    lexicon_version,
    parse_lexicon,
    pattern_source,
)

LEXICON = """
# Westerosi names read with Hungarian letter values
Westeros = Veszterosz
Aegon = Egon
Alys = Álisz
Alyssa = Álissza
"""


class ParseLexiconTest(unittest.TestCase):
    def test_comments_and_blank_lines_are_ignored(self):
        self.assertEqual(
            parse_lexicon(LEXICON),
            [
                ('Westeros', 'Veszterosz'),
                ('Aegon', 'Egon'),
                ('Alys', 'Álisz'),
                ('Alyssa', 'Álissza'),
            ],
        )

    def test_alternative_separators(self):
        self.assertEqual(
            parse_lexicon('Aegon -> Egon\nJaehaerys => Dzsehérisz\nRhaenyra → Renira'),
            [('Aegon', 'Egon'), ('Jaehaerys', 'Dzsehérisz'), ('Rhaenyra', 'Renira')],
        )

    def test_half_typed_rules_are_skipped(self):
        self.assertEqual(parse_lexicon('Westeros =\n= Veszterosz\nno separator'), [])

    def test_empty_lexicon(self):
        self.assertEqual(parse_lexicon(''), [])
        self.assertEqual(parse_lexicon(None), [])


class ApplyPronunciationTest(unittest.TestCase):
    def test_plain_replacement(self):
        self.assertEqual(
            apply_pronunciation('Westeros históriája', LEXICON),
            'Veszterosz históriája',
        )

    def test_hungarian_suffix_is_kept(self):
        self.assertEqual(
            apply_pronunciation('Westerosban és Aegonnak', LEXICON),
            'Veszteroszban és Egonnak',
        )

    def test_hyphenated_suffix_is_kept(self):
        self.assertEqual(
            apply_pronunciation('Aegon-nak adta', LEXICON),
            'Egon-nak adta',
        )

    def test_longer_rule_wins(self):
        self.assertEqual(
            apply_pronunciation('Alyssa és Alys', LEXICON),
            'Álissza és Álisz',
        )

    def test_capitalization_is_carried_over(self):
        self.assertEqual(
            apply_pronunciation('WESTEROS, Westeros, westeros', LEXICON),
            'VESZTEROSZ, Veszterosz, veszterosz',
        )

    def test_no_match_inside_another_word(self):
        self.assertEqual(
            apply_pronunciation('Nyugat-Westeros és XWesteros', LEXICON),
            'Nyugat-Veszterosz és XWesteros',
        )

    def test_bracket_tags_and_control_tokens_survive(self):
        self.assertEqual(
            apply_pronunciation('[laugh] Aegon <|emotion:joy|> Westeros', LEXICON),
            '[laugh] Egon <|emotion:joy|> Veszterosz',
        )

    def test_empty_lexicon_is_a_no_op(self):
        self.assertEqual(apply_pronunciation('Westeros', ''), 'Westeros')

    def test_empty_text_is_a_no_op(self):
        self.assertEqual(apply_pronunciation('', LEXICON), '')


class EditorDataTest(unittest.TestCase):
    def test_entries_expose_both_sides_of_each_rule(self):
        self.assertEqual(
            entries('Westeros = Veszterosz\nAegon = Egon'),
            [
                {'source': 'Westeros', 'spoken': 'Veszterosz'},
                {'source': 'Aegon', 'spoken': 'Egon'},
            ],
        )

    def test_pattern_source_matches_what_the_engine_rewrites(self):
        raw = 'Westeros = Veszterosz'
        pattern = re.compile(pattern_source(raw), re.IGNORECASE)

        # The reader marks with this pattern, so it has to agree with the
        # rewrite on suffixes, casing and word boundaries.
        self.assertTrue(pattern.search('Westerosban'))
        self.assertTrue(pattern.search('WESTEROS'))
        self.assertFalse(pattern.search('XWesteros'))
        self.assertEqual(pattern.search('Nyugat-Westeros').group(1), 'Westeros')

    def test_pattern_source_is_empty_without_rules(self):
        self.assertEqual(pattern_source(''), '')


class LexiconVersionTest(unittest.TestCase):
    def test_editing_a_rule_changes_the_version(self):
        self.assertNotEqual(
            lexicon_version('Westeros = Veszterosz'),
            lexicon_version('Westeros = Vesztirosz'),
        )

    def test_comments_and_order_do_not_change_audio(self):
        self.assertEqual(
            lexicon_version('# comment\nWesteros = Veszterosz\n'),
            lexicon_version('Westeros = Veszterosz'),
        )

    def test_sorting_the_rules_does_not_change_audio(self):
        # Match precedence comes from rule length, so alphabetizing the list
        # must not invalidate a book's synthesized audio.
        self.assertEqual(
            lexicon_version('Westeros = Veszterosz\nAegon = Egon'),
            lexicon_version('Aegon = Egon\nWesteros = Veszterosz'),
        )

    def test_empty_lexicon_has_a_stable_version(self):
        self.assertEqual(lexicon_version(''), '0')


if __name__ == '__main__':
    unittest.main()
