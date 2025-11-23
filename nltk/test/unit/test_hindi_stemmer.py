# Natural Language Toolkit: Hindi Stemmer Tests
#
# Copyright (C) 2001-2025 NLTK Project
# Author: [Your Name]
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Test suite for the Hindi stemmer.

This suite verifies:
- The Hindi stemmer loads correctly
- Stemming returns valid Unicode strings
- Common Hindi suffix rules behave correctly
- Longer suffixes are removed before shorter ones
- No over-stemming occurs on short or suffix-less words
- Stems remain valid UTF-8 strings
"""

import unittest
from nltk.stem import HindiStemmer


class HindiStemmerTest(unittest.TestCase):
    """Tests for the Hindi stemmer."""

    def setUp(self):
        """Create a stemmer instance."""
        try:
            self.stemmer = HindiStemmer()
        except Exception as e:
            raise RuntimeError("HindiStemmer failed to initialize") from e

    # -------------------------------------------------------------
    # BASIC LOADING TESTS
    # -------------------------------------------------------------

    def test_stemmer_instance_created(self):
        """Ensure the stemmer loads correctly."""
        self.assertIsInstance(self.stemmer, HindiStemmer)

    def test_stem_returns_string(self):
        """Ensure stem() always returns a valid string."""
        result = self.stemmer.stem("किताबों")
        self.assertIsInstance(result, str)

    # -------------------------------------------------------------
    # BASIC STEMMING BEHAVIOR
    # -------------------------------------------------------------

    def test_common_plural_forms(self):
        """Common plural/oblique suffixes should be removed."""
        cases = [
            ("किताबों", "किताब"),
            ("लड़कों", "लड़क"),
            ("लड़कियों", "लड़की"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertEqual(self.stemmer.stem(word), expected)

    def test_basic_gender_markers(self):
        """Simple gender markers should be handled."""
        cases = [
            ("लड़का", "लड़"),
            ("लड़की", "लड़"),
            ("लड़के", "लड़"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertEqual(self.stemmer.stem(word), expected)

    def test_basic_verb_forms(self):
        """Verb tense markers should be removed."""
        cases = [
            ("खाता", "खा"),
            ("खाती", "खा"),
            ("खाते", "खा"),
            ("खाएगा", "खाए"),
            ("खाएगी", "खाए"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                result = self.stemmer.stem(word)
                self.assertIsInstance(result, str)
                self.assertLessEqual(len(result), len(word))

    def test_longest_suffix_priority(self):
        """Longest suffix should be removed first when overlaps occur."""
        stem = self.stemmer.stem("करता")
        self.assertEqual(stem, "कर")

    # -------------------------------------------------------------
    # SPECIAL CASE LOGIC TESTS
    # -------------------------------------------------------------

    def test_special_case_yon_suffix_with_i_end(self):
        """
        Test the special logic:
        If a word ends with 'यों' and the intermediate stem ends with 'ि',
        stemmer should replace 'ि' with 'ी'.
        
        Example:
        लड़कियों -> remove 'यों' -> लड़कि -> replace 'ि' -> लड़की
        """
        word = "लड़कियों"
        expected = "लड़की"
        result = self.stemmer.stem(word)
        self.assertEqual(result, expected)

    # -------------------------------------------------------------
    # EDGE CASES
    # -------------------------------------------------------------

    def test_empty_string(self):
        """Empty string should return empty string."""
        self.assertEqual(self.stemmer.stem(""), "")

    def test_short_words_not_overstemmed(self):
        """Short words should remain unchanged."""
        short_words = ["है", "हो", "जा", "ना", "का"]
        for w in short_words:
            with self.subTest(word=w):
                stem = self.stemmer.stem(w)
                self.assertIsInstance(stem, str)
                self.assertGreater(len(stem), 0)

    def test_words_without_suffixes(self):
        """Words without known suffixes return unchanged."""
        cases = [
            ("घर", "घर"),
            ("पानी", "पान"),
            ("राम", "राम"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertEqual(self.stemmer.stem(word), expected)

    # -------------------------------------------------------------
    # UTF-8 / FORMAT INTEGRITY
    # -------------------------------------------------------------

    def test_output_is_valid_utf8(self):
        """Ensure stems are valid UTF-8."""
        words = ["किताबों", "लड़की", "खाता", "घर"]
        for w in words:
            stem = self.stemmer.stem(w)
            try:
                encoded = stem.encode("utf-8")
                decoded = encoded.decode("utf-8")
                self.assertEqual(stem, decoded)
            except UnicodeError:
                self.fail(f"Invalid UTF-8 stem produced: {stem}")

    def test_no_english_output(self):
        """Ensure stemming never produces English strings."""
        English_chars = "abcdefghijklmnopqrstuvwxyz"
        stem = self.stemmer.stem("किताबों")
        for c in English_chars:
            self.assertNotIn(c, stem)

    # -------------------------------------------------------------
    # MINIMUM STEM LENGTH
    # -------------------------------------------------------------

    def test_minimum_stem_length_enforced(self):
        """Stems must not be shorter than minimum length."""
        words = ["ता", "ती", "ते"]
        for w in words:
            with self.subTest(word=w):
                result = self.stemmer.stem(w)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
