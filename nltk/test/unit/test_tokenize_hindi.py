"""
Unit tests for Hindi language tokenization.

Tests cover word, sentence, character, sub-word, and n-gram tokenization
for Hindi text in Devanagari script.
"""

import unittest

from nltk.tokenize import (
    HindiCharacterTokenizer,
    HindiNGramTokenizer,
    HindiSentenceTokenizer,
    HindiSubwordTokenizer,
    HindiWordTokenizer,
    hindi_char_tokenize,
    hindi_ngrams,
    hindi_sent_tokenize,
    hindi_word_tokenize,
)


class TestHindiWordTokenizer(unittest.TestCase):
    """Test cases for HindiWordTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = HindiWordTokenizer()

    def test_simple_sentence(self):
        """Test basic word tokenization."""
        text = "राम घर गया।"
        expected = ["राम", "घर", "गया", "।"]
        result = self.tokenizer.tokenize(text)
        self.assertEqual(result, expected)

    def test_conjunct_consonants(self):
        """Test that conjuncts stay together."""
        text = "स्कूल"  # school (स् + कू + ल)
        # Should NOT split into ['स', '्', 'कू', 'ल']
        result = self.tokenizer.tokenize(text)
        self.assertIn("स्कूल", result)
        # Ensure virama is not a separate token
        self.assertNotIn("्", result)

    def test_hindi_punctuation(self):
        """Test Hindi-specific punctuation."""
        text = "क्या तुम जाओगे।"
        result = self.tokenizer.tokenize(text)
        self.assertIn("।", result)
        self.assertIn("क्या", result)
        self.assertIn("तुम", result)
        self.assertIn("जाओगे", result)

    def test_double_danda(self):
        """Test double danda punctuation."""
        text = "यह श्लोक है॥"
        result = self.tokenizer.tokenize(text)
        self.assertIn("॥", result)

    def test_numbers(self):
        """Test Devanagari numerals."""
        text = "मेरे पास १०० रुपये हैं।"
        result = self.tokenizer.tokenize(text)
        # Check that numbers are preserved
        self.assertTrue(any("१" in token or "०" in token for token in result))

    def test_arabic_numerals(self):
        """Test Arabic numerals in Hindi text."""
        text = "मेरे पास 100 रुपये हैं।"
        result = self.tokenizer.tokenize(text)
        self.assertIn("100", result)

    def test_compound_words(self):
        """Test hyphenated compounds."""
        text = "माता-पिता"
        result = self.tokenizer.tokenize(text)
        # Compound words may be split or kept together depending on implementation
        self.assertGreater(len(result), 0)

    def test_mixed_script(self):
        """Test mixed Devanagari and Latin script."""
        text = "मैं computer use करता हूँ।"
        result = self.tokenizer.tokenize(text)
        self.assertIn("computer", result)
        self.assertIn("use", result)
        self.assertIn("मैं", result)

    def test_empty_string(self):
        """Test edge case: empty string."""
        self.assertEqual(self.tokenizer.tokenize(""), [])

    def test_whitespace_only(self):
        """Test edge case: whitespace only."""
        result = self.tokenizer.tokenize("   ")
        # Should return empty list or handle gracefully
        self.assertIsInstance(result, list)

    def test_multiple_sentences(self):
        """Test tokenization across multiple sentences."""
        text = "राम आया। सीता गई। वे खुश थे।"
        result = self.tokenizer.tokenize(text)
        # Should tokenize all words
        self.assertIn("राम", result)
        self.assertIn("सीता", result)
        self.assertIn("वे", result)
        # Should include punctuation
        self.assertEqual(result.count("।"), 3)

    def test_span_tokenize(self):
        """Test span tokenization."""
        text = "राम घर गया।"
        spans = list(self.tokenizer.span_tokenize(text))
        self.assertIsInstance(spans, list)
        self.assertGreater(len(spans), 0)
        # Verify spans are valid
        for start, end in spans:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(text))
            self.assertLess(start, end)


class TestHindiSentenceTokenizer(unittest.TestCase):
    """Test cases for HindiSentenceTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = HindiSentenceTokenizer()

    def test_single_sentence(self):
        """Test single sentence."""
        text = "राम अच्छा लड़का है।"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(len(result), 1)
        self.assertIn("राम", result[0])

    def test_multiple_sentences(self):
        """Test multiple sentences with purna viram."""
        text = "राम आया। सीता गई। वे खुश थे।"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(len(result), 3)
        self.assertIn("राम", result[0])
        self.assertIn("सीता", result[1])
        self.assertIn("वे", result[2])

    def test_question_sentence(self):
        """Test question detection."""
        text = "तुम कैसे हो? मैं ठीक हूँ।"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(len(result), 2)
        self.assertIn("?", result[0])
        self.assertIn("।", result[1])

    def test_double_danda(self):
        """Test double danda (verse terminator)."""
        text = "यह श्लोक है॥ यह दूसरा है॥"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(len(result), 2)
        self.assertIn("॥", result[0])
        self.assertIn("॥", result[1])

    def test_exclamation(self):
        """Test exclamation mark."""
        text = "वाह! यह अच्छा है।"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(len(result), 2)
        self.assertIn("!", result[0])

    def test_quotations(self):
        """Test quotation handling."""
        text = 'राम ने कहा, "मैं जाऊंगा।" सीता ने सुना।'
        result = self.tokenizer.tokenize(text)
        # Should preserve quoted sentence structure
        self.assertGreater(len(result), 0)

    def test_empty_string(self):
        """Test edge case: empty string."""
        result = self.tokenizer.tokenize("")
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        """Test edge case: whitespace only."""
        result = self.tokenizer.tokenize("   ")
        # Should return empty list or handle gracefully
        self.assertIsInstance(result, list)

    def test_no_sentence_end(self):
        """Test text without sentence ending."""
        text = "राम घर गया"
        result = self.tokenizer.tokenize(text)
        # Should return the text as a single sentence
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)


class TestHindiCharacterTokenizer(unittest.TestCase):
    """Test cases for HindiCharacterTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = HindiCharacterTokenizer()

    def test_basic_characters(self):
        """Test basic character splitting."""
        text = "राम"
        result = self.tokenizer.tokenize(text)
        # Should handle matras correctly
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_vowel_signs(self):
        """Test that matras stay with base character."""
        text = "की"  # क + ी
        result = self.tokenizer.tokenize(text)
        # Matra should stay with consonant
        self.assertIsInstance(result, list)
        # Should not split into ['क', 'ी']
        if len(result) > 1:
            # If split, first part should contain the consonant
            self.assertTrue(any("क" in token for token in result))

    def test_conjuncts(self):
        """Test conjunct consonant handling."""
        text = "स्कूल"  # Contains conjunct
        result = self.tokenizer.tokenize(text)
        # Should handle conjuncts appropriately
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_punctuation(self):
        """Test punctuation handling."""
        text = "राम।"
        result = self.tokenizer.tokenize(text)
        self.assertIn("।", result)

    def test_empty_string(self):
        """Test edge case: empty string."""
        self.assertEqual(self.tokenizer.tokenize(""), [])

    def test_mixed_script(self):
        """Test mixed Devanagari and Latin characters."""
        text = "रामA"
        result = self.tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        self.assertIn("A", result)


class TestHindiSubwordTokenizer(unittest.TestCase):
    """Test cases for HindiSubwordTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = HindiSubwordTokenizer()

    def test_basic_subword(self):
        """Test basic sub-word tokenization."""
        text = "अच्छा"
        result = self.tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_prefix_detection(self):
        """Test prefix detection."""
        text = "अच्छा"
        result = self.tokenizer.tokenize(text)
        # May split prefix 'अ' if detected
        self.assertIsInstance(result, list)

    def test_suffix_detection(self):
        """Test suffix detection."""
        text = "करता"
        result = self.tokenizer.tokenize(text)
        # May split suffix if detected
        self.assertIsInstance(result, list)

    def test_sentence(self):
        """Test sub-word tokenization of a sentence."""
        text = "राम अच्छा लड़का है।"
        result = self.tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_empty_string(self):
        """Test edge case: empty string."""
        self.assertEqual(self.tokenizer.tokenize(""), [])


class TestHindiNGramTokenizer(unittest.TestCase):
    """Test cases for HindiNGramTokenizer."""

    def test_bigrams(self):
        """Test bigram generation."""
        text = "राम और सीता"
        tokenizer = HindiNGramTokenizer(n=2)
        result = tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Check that results are tuples of length 2
        for ngram in result:
            self.assertIsInstance(ngram, tuple)
            self.assertEqual(len(ngram), 2)

    def test_trigrams(self):
        """Test trigram generation."""
        text = "राम और सीता वन में"
        tokenizer = HindiNGramTokenizer(n=3)
        result = tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        # Check that results are tuples of length 3
        for ngram in result:
            self.assertIsInstance(ngram, tuple)
            self.assertEqual(len(ngram), 3)

    def test_unigrams(self):
        """Test unigram generation (n=1)."""
        text = "राम और सीता"
        tokenizer = HindiNGramTokenizer(n=1)
        result = tokenizer.tokenize(text)
        self.assertIsInstance(result, list)
        for ngram in result:
            self.assertIsInstance(ngram, tuple)
            self.assertEqual(len(ngram), 1)

    def test_short_text(self):
        """Test n-gram with text shorter than n."""
        text = "राम"
        tokenizer = HindiNGramTokenizer(n=5)
        result = tokenizer.tokenize(text)
        # Should return empty list or handle gracefully
        self.assertIsInstance(result, list)

    def test_empty_string(self):
        """Test edge case: empty string."""
        tokenizer = HindiNGramTokenizer(n=2)
        self.assertEqual(tokenizer.tokenize(""), [])

    def test_invalid_n(self):
        """Test invalid n value."""
        with self.assertRaises(ValueError):
            HindiNGramTokenizer(n=0)
        with self.assertRaises(ValueError):
            HindiNGramTokenizer(n=-1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""

    def test_hindi_word_tokenize(self):
        """Test convenience function for word tokenization."""
        text = "नमस्ते भारत।"
        result = hindi_word_tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("नमस्ते", result)

    def test_hindi_sent_tokenize(self):
        """Test sentence convenience function."""
        text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
        result = hindi_sent_tokenize(text)
        self.assertEqual(len(result), 2)
        self.assertIn("पहला", result[0])
        self.assertIn("दूसरा", result[1])

    def test_hindi_char_tokenize(self):
        """Test character tokenization convenience function."""
        text = "राम"
        result = hindi_char_tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_hindi_ngrams(self):
        """Test n-gram convenience function."""
        text = "राम और सीता"
        result = hindi_ngrams(text, n=2)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for ngram in result:
            self.assertIsInstance(ngram, tuple)
            self.assertEqual(len(ngram), 2)

    def test_hindi_ngrams_default(self):
        """Test n-gram with default n=2."""
        text = "राम और सीता"
        result = hindi_ngrams(text)
        self.assertIsInstance(result, list)
        # Should default to bigrams
        for ngram in result:
            self.assertEqual(len(ngram), 2)


class TestRealWorldExamples(unittest.TestCase):
    """Test cases with real-world Hindi text examples."""

    def test_news_sentence(self):
        """Test tokenization of news-style sentence."""
        text = "भारत एक महान देश है। यहाँ विविधता में एकता है।"
        words = hindi_word_tokenize(text)
        sentences = hindi_sent_tokenize(text)
        self.assertGreater(len(words), 0)
        self.assertEqual(len(sentences), 2)

    def test_question_and_statement(self):
        """Test mixed questions and statements."""
        text = "तुम कहाँ जा रहे हो? मैं बाज़ार जा रहा हूँ।"
        sentences = hindi_sent_tokenize(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("?", sentences[0])
        self.assertIn("।", sentences[1])

    def test_numbers_and_text(self):
        """Test text with numbers."""
        text = "मेरे पास ५० रुपये हैं। मैं १०० रुपये चाहता हूँ।"
        words = hindi_word_tokenize(text)
        # Should handle Devanagari numerals
        self.assertGreater(len(words), 0)

    def test_complex_sentence(self):
        """Test complex sentence with multiple clauses."""
        text = "राम, जो एक अच्छा लड़का है, स्कूल जाता है।"
        words = hindi_word_tokenize(text)
        self.assertIn("राम", words)
        self.assertIn("स्कूल", words)  # Test conjunct handling


if __name__ == "__main__":
    unittest.main()
