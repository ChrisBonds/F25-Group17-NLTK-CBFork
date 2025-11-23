# Natural Language Toolkit: Hindi Stopwords Tests
#
# Copyright (C) 2001-2025 NLTK Project
# Author: [Your Name]
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Test suite for the Hindi stopwords corpus.

This suite verifies:
- The Hindi stopwords file loads correctly using NLTK's StopwordsCorpusReader
- The stopwords list is non-empty
- Expected common Hindi stopwords are present
- All stopwords are valid Unicode strings
- No blank lines or whitespace artifacts exist
"""

import unittest
from nltk.corpus import stopwords


class HindiStopwordsTest(unittest.TestCase):
    """Tests for the Hindi stopwords list."""

    def setUp(self):
        """Load Hindi stopwords list."""
        try:
            self.hindi_stops = stopwords.words("hindi")
        except OSError:
            raise OSError(
                "Hindi stopwords not found. Ensure `nltk_data/corpora/stopwords/hindi` exists."
            )

    # -------------------------------------------------------------
    # BASIC LOADING TESTS
    # -------------------------------------------------------------

    def test_hindi_stopwords_load(self):
        """Ensure the Hindi stopwords list loads successfully."""
        self.assertIsInstance(self.hindi_stops, list)
        self.assertGreater(len(self.hindi_stops), 0)

    def test_hindi_stopwords_are_strings(self):
        """Ensure all stopwords are valid Unicode strings."""
        for word in self.hindi_stops:
            self.assertIsInstance(word, str)
            self.assertGreater(len(word.strip()), 0)

    # -------------------------------------------------------------
    # CONTENT VALIDATION
    # -------------------------------------------------------------

    def test_common_hindi_stopwords_present(self):
        """Verify presence of common Hindi stopwords."""
        expected_words = [
            "और",
            "है",
            "था",
            "यह",
            "वह",
            "के",
            "को",
            "में",
            "से",
            "पर",
            "ही",
            "भी",
        ]

        for word in expected_words:
            with self.subTest(word=word):
                self.assertIn(word, self.hindi_stops)

    def test_no_duplicates(self):
        """Ensure the stopword list contains no duplicates."""
        self.assertEqual(len(self.hindi_stops), len(set(self.hindi_stops)))

    # -------------------------------------------------------------
    # FORMAT INTEGRITY
    # -------------------------------------------------------------

    def test_no_empty_lines_or_whitespace(self):
        """Ensure no blank or whitespace-only entries exist."""
        for word in self.hindi_stops:
            stripped = word.strip()
            self.assertGreater(len(stripped), 0)
            self.assertEqual(word, stripped)

    def test_no_english_words_present(self):
        """Ensure common English stopwords are NOT mistakenly included."""
        english_words = ["the", "and", "is", "for", "this"]

        for word in english_words:
            self.assertNotIn(word, self.hindi_stops)

    # -------------------------------------------------------------
    # ROUND TRIP CONSISTENCY
    # -------------------------------------------------------------

    def test_stopwords_in_lowercase(self):
        """Ensure all entries are in consistent lowercase (Hindi has no uppercase)."""
        for word in self.hindi_stops:
            self.assertEqual(word, word.lower())

    def test_encoding_is_valid_utf8(self):
        """Verify stopwords appear to be valid UTF-8 Hindi characters."""
        for word in self.hindi_stops:
            try:
                encoded = word.encode("utf-8")
                decoded = encoded.decode("utf-8")
                self.assertEqual(word, decoded)
            except UnicodeError:
                self.fail(f"Invalid UTF-8 encoding detected in word: {word}")


if __name__ == "__main__":
    unittest.main()
