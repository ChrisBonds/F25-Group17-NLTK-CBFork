# Natural Language Toolkit: Hindi Language Tokenizers
#
# Copyright (C) 2001-2025 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Hindi language tokenization support.

This module provides tokenization functionality for Hindi text written in
Devanagari script. It includes word, sentence, character, sub-word, and
n-gram tokenization capabilities.

Devanagari Script Characteristics:
- Unicode range: U+0900 to U+097F (Devanagari)
- Unicode range: U+A8E0 to U+A8FF (Devanagari Extended)
- Vowels: अ-औ (U+0905-U+0914)
- Consonants: क-ह (U+0915-U+0939)
- Vowel signs (matras): ा-ौ (U+093E-U+094C)
- Virama (halant): ् (U+094D)
- Hindi punctuation: । (purna viram), ॥ (double danda)
- Devanagari numerals: ०-९ (U+0966-U+096F)

Example usage:

    >>> from nltk.tokenize import HindiWordTokenizer, hindi_word_tokenize
    >>> text = "राम और सीता वन में गए।"
    >>> tokenizer = HindiWordTokenizer()
    >>> tokenizer.tokenize(text)
    ['राम', 'और', 'सीता', 'वन', 'में', 'गए', '।']
    >>> hindi_word_tokenize(text)
    ['राम', 'और', 'सीता', 'वन', 'में', 'गए', '।']
"""

import re
from typing import Iterator, List, Tuple

from nltk.tokenize.indic import (
    IndicSentenceTokenizer,
    IndicTokenizer,
    IndicWordTokenizer,
)
from nltk.tokenize.util import align_tokens

# Unicode ranges for Devanagari script
DEVANAGARI_BASE = r"\u0900-\u097F"
DEVANAGARI_EXTENDED = r"\uA8E0-\uA8FF"
DEVANAGARI_ALL = f"[{DEVANAGARI_BASE}{DEVANAGARI_EXTENDED}]"

# Devanagari characters
DEVANAGARI_VOWELS = r"\u0905-\u0914"  # अ-औ
DEVANAGARI_CONSONANTS = r"\u0915-\u0939"  # क-ह
DEVANAGARI_VOWEL_SIGNS = r"\u093E-\u094C"  # ा-ौ
DEVANAGARI_VIrama = r"\u094D"  # ् (halant)
DEVANAGARI_NUMERALS = r"\u0966-\u096F"  # ०-९

# Hindi-specific punctuation
HINDI_PURNA_VIram = r"\u0964"  # ।
HINDI_DOUBLE_DANDA = r"\u0965"  # ॥

# Pattern for Devanagari word (consonant + optional matras + optional virama + consonant)
DEVANAGARI_WORD_PATTERN = (
    rf"[{DEVANAGARI_CONSONANTS}][{DEVANAGARI_VOWEL_SIGNS}]*"
    rf"({DEVANAGARI_VIrama}[{DEVANAGARI_CONSONANTS}][{DEVANAGARI_VOWEL_SIGNS}]*)*"
    rf"|[{DEVANAGARI_VOWELS}]"
    rf"|[{DEVANAGARI_NUMERALS}]+"
)


class HindiWordTokenizer(IndicWordTokenizer):
    """
    Hindi-specific word tokenization.

    Handles Devanagari script word boundaries, Hindi punctuation marks,
    compound words, and conjunct consonants. Keeps conjuncts together
    (doesn't split at virama).

    Example:

        >>> tokenizer = HindiWordTokenizer()
        >>> text = "राम और सीता वन में गए।"
        >>> tokenizer.tokenize(text)
        ['राम', 'और', 'सीता', 'वन', 'में', 'गए', '।']
        >>> text = "स्कूल"  # school (conjunct: स् + कू + ल)
        >>> tokenizer.tokenize(text)
        ['स्कूल']
    """

    def __init__(self):
        """Initialize the Hindi word tokenizer."""
        # Pattern to match Devanagari words, numbers, and punctuation
        # This pattern matches:
        # - Devanagari words (consonants with optional matras and conjuncts)
        # - Devanagari vowels
        # - Devanagari numerals
        # - Hindi punctuation (।, ॥)
        # - Western punctuation
        # - Latin characters (for loanwords)
        # - Arabic numerals
        # Build pattern: match words first, then punctuation
        # Order matters - more specific patterns first
        self._word_pattern = re.compile(
            rf"{DEVANAGARI_WORD_PATTERN}+"  # One or more Devanagari words
            rf"|[{DEVANAGARI_NUMERALS}]+"  # Devanagari numerals
            rf"|[a-zA-Z0-9]+"  # Latin words and Arabic numerals
            rf"|[{HINDI_PURNA_VIram}{HINDI_DOUBLE_DANDA}?!.,;:()\[\]{{}}'\"\\-]"  # Punctuation
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Return a tokenized copy of text, split into words.

        :param text: Input Hindi text in Devanagari script
        :type text: str
        :return: List of word tokens
        :rtype: List[str]

        Example:

            >>> tokenizer = HindiWordTokenizer()
            >>> tokenizer.tokenize("राम घर गया।")
            ['राम', 'घर', 'गया', '।']
        """
        if not text or not text.strip():
            return []

        # Normalize text: remove zero-width characters and normalize whitespace
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)  # Remove zero-width chars
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace

        # Split on whitespace, then separate punctuation from words
        tokens = []
        words = text.split()

        for word in words:
            if not word:
                continue

            # Find the boundary between word and punctuation
            # Check from the end: if a character is not Devanagari and not alphanumeric,
            # it's likely punctuation
            word_end = len(word)

            for i in range(len(word) - 1, -1, -1):
                char = word[i]
                char_code = ord(char)

                # Check for Hindi punctuation first (before Devanagari check)
                # U+0964 = । (purna viram), U+0965 = ॥ (double danda)
                is_hindi_punct = char_code == 0x0964 or char_code == 0x0965
                # Check for other punctuation
                is_other_punct = char in "?!.,;:()[]{}'\"-"

                if is_hindi_punct or is_other_punct:
                    # This is punctuation, continue looking backwards for word end
                    continue

                # Check if it's Devanagari (base or extended range, excluding punctuation)
                # Note: U+0964 and U+0965 are in Devanagari range but are punctuation
                is_devanagari = (0x0900 <= char_code <= 0x097F) or (
                    0xA8E0 <= char_code <= 0xA8FF
                )
                # Check if it's alphanumeric (Latin/Arabic)
                is_alphanumeric = (
                    ("a" <= char <= "z") or ("A" <= char <= "Z") or ("0" <= char <= "9")
                )

                if is_devanagari or is_alphanumeric:
                    # Found the end of the word part
                    word_end = i + 1
                    break
                # Unknown character, treat as punctuation

            # Split word and punctuation
            if word_end < len(word):
                word_part = word[:word_end]
                punct_part = word[word_end:]
                if word_part:
                    tokens.append(word_part)
                # Add each punctuation character separately
                for punct_char in punct_part:
                    tokens.append(punct_char)
            else:
                # No punctuation found, add whole word
                tokens.append(word)

        return tokens

    def span_tokenize(self, text: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets (start_i, end_i),
        where text[start_i:end_i] is the corresponding token.

        :param text: Input Hindi text
        :type text: str
        :rtype: Iterator[Tuple[int, int]]
        """
        tokens = self.tokenize(text)
        yield from align_tokens(tokens, text)


class HindiSentenceTokenizer(IndicSentenceTokenizer):
    """
    Hindi-specific sentence tokenization.

    Handles Devanagari sentence boundary detection considering:
    - Devanagari sentence terminators (।, ॥)
    - Question marks and exclamation marks
    - Quotation handling
    - Abbreviation detection

    Example:

        >>> tokenizer = HindiSentenceTokenizer()
        >>> text = "राम आया। सीता गई। वे खुश थे।"
        >>> tokenizer.tokenize(text)
        ['राम आया।', 'सीता गई।', 'वे खुश थे।']
    """

    def __init__(self):
        """Initialize the Hindi sentence tokenizer."""
        # Sentence terminators: purna viram, double danda, question mark, exclamation
        self._sentence_endings = re.compile(
            rf"[{HINDI_PURNA_VIram}{HINDI_DOUBLE_DANDA}?!।॥]+"
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Return a tokenized copy of text, split into sentences.

        :param text: Input Hindi text in Devanagari script
        :type text: str
        :return: List of sentence strings
        :rtype: List[str]

        Example:

            >>> tokenizer = HindiSentenceTokenizer()
            >>> text = "राम आया। सीता गई।"
            >>> tokenizer.tokenize(text)
            ['राम आया।', 'सीता गई।']
        """
        if not text or not text.strip():
            return []

        # Normalize text
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        text = text.strip()

        # Split on sentence boundaries
        sentences = []
        current_sentence = ""
        i = 0

        while i < len(text):
            char = text[i]
            current_sentence += char

            # Check if this is a sentence terminator
            if self._sentence_endings.match(char):
                # Look ahead to see if there's more punctuation
                j = i + 1
                while j < len(text) and self._sentence_endings.match(text[j]):
                    current_sentence += text[j]
                    j += 1

                # Check if next character is whitespace or end of string
                if j >= len(text) or text[j].isspace():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    # Skip whitespace after sentence end
                    while j < len(text) and text[j].isspace():
                        j += 1
                    i = j
                    continue

            i += 1

        # Add remaining text as a sentence if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences if sentences else [text]


class HindiCharacterTokenizer(IndicTokenizer):
    """
    Character-level tokenization for Hindi text.

    Handles Unicode Devanagari character boundaries, keeping vowel signs
    (matras) with their base consonants and preserving conjunct consonants.

    Example:

        >>> tokenizer = HindiCharacterTokenizer()
        >>> tokenizer.tokenize("राम")
        ['रा', 'म']  # र + ा (matra) stays together, then म
    """

    def tokenize(self, text: str) -> List[str]:
        """
        Return a tokenized copy of text, split into characters or character clusters.

        Character clusters include:
        - Base consonant + vowel sign (matra)
        - Conjunct consonants (consonant + virama + consonant)
        - Standalone vowels

        :param text: Input Hindi text
        :type text: str
        :return: List of character tokens
        :rtype: List[str]

        Example:

            >>> tokenizer = HindiCharacterTokenizer()
            >>> tokenizer.tokenize("की")
            ['की']  # क + ी (matra) stays together
        """
        if not text:
            return []

        # Normalize text
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        characters = []
        i = 0

        while i < len(text):
            char = text[i]
            code_point = ord(char)

            # Check if it's a Devanagari character
            if 0x0900 <= code_point <= 0x097F or 0xA8E0 <= code_point <= 0xA8FF:
                # Start of a character cluster
                cluster = char
                i += 1

                # Collect combining marks (matras, virama, etc.)
                while i < len(text):
                    next_code = ord(text[i])
                    # Vowel signs (matras): U+093E-U+094C
                    # Virama: U+094D
                    # Other combining marks: U+0900-U+0939 (various marks)
                    if (
                        (0x093E <= next_code <= 0x094C)  # Matras
                        or next_code == 0x094D  # Virama
                        or (0x0900 <= next_code <= 0x0939)  # Other combining marks
                    ):
                        cluster += text[i]
                        i += 1
                    else:
                        break

                characters.append(cluster)
            else:
                # Non-Devanagari character (punctuation, Latin, etc.)
                characters.append(char)
                i += 1

        return characters


class HindiSubwordTokenizer(IndicTokenizer):
    """
    Sub-word tokenization for morphologically rich Hindi.

    Implements a simple rule-based segmentation approach that identifies
    common prefixes and suffixes in Hindi words. This is a basic implementation
    that can be extended with more sophisticated methods.

    Example:

        >>> tokenizer = HindiSubwordTokenizer()
        >>> tokenizer.tokenize("राजनीतिक")
        ['राजनीतिक']  # Basic implementation - can be enhanced
    """

    # Common Hindi prefixes and suffixes (basic list)
    COMMON_PREFIXES = ["अ", "अन", "अप", "अव", "उप", "प्र", "पर", "सु", "दु"]
    COMMON_SUFFIXES = [
        "ता",
        "ती",
        "ते",
        "ना",
        "नी",
        "ने",
        "कर",
        "करा",
        "करी",
        "वाला",
        "वाली",
        "वाले",
    ]

    def __init__(self):
        """Initialize the Hindi sub-word tokenizer."""
        # Compile prefix and suffix patterns
        self._prefix_pattern = re.compile(
            "^(" + "|".join(re.escape(p) for p in self.COMMON_PREFIXES) + ")"
        )
        self._suffix_pattern = re.compile(
            "(" + "|".join(re.escape(s) for s in self.COMMON_SUFFIXES) + ")$"
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Return a tokenized copy of text, split into sub-words.

        This is a basic implementation that attempts to identify
        prefixes and suffixes. More sophisticated methods (e.g., BPE)
        can be added in the future.

        :param text: Input Hindi text (word or sentence)
        :type text: str
        :return: List of sub-word tokens
        :rtype: List[str]

        Example:

            >>> tokenizer = HindiSubwordTokenizer()
            >>> tokenizer.tokenize("अच्छा")
            ['अ', 'च्छा']  # Prefix 'अ' separated
        """
        if not text or not text.strip():
            return []

        # First, tokenize into words
        word_tokenizer = HindiWordTokenizer()
        words = word_tokenizer.tokenize(text)

        subwords = []
        for word in words:
            # Skip punctuation
            if not re.match(rf"^{DEVANAGARI_WORD_PATTERN}$", word):
                subwords.append(word)
                continue

            # Try to split prefix
            prefix_match = self._prefix_pattern.match(word)
            if prefix_match:
                prefix = prefix_match.group(1)
                remaining = word[len(prefix) :]
                if remaining:
                    subwords.append(prefix)
                    word = remaining

            # Try to split suffix
            suffix_match = self._suffix_pattern.search(word)
            if suffix_match:
                suffix = suffix_match.group(1)
                root = word[: -len(suffix)]
                if root:
                    subwords.append(root)
                    subwords.append(suffix)
                else:
                    subwords.append(word)
            else:
                subwords.append(word)

        return subwords


class HindiNGramTokenizer(IndicTokenizer):
    """
    N-gram generation for Hindi text.

    Generates n-grams from tokenized Hindi text. Can work at word level
    or character level.

    Example:

        >>> tokenizer = HindiNGramTokenizer(n=2)
        >>> text = "राम और सीता"
        >>> tokenizer.tokenize(text)
        [('राम', 'और'), ('और', 'सीता')]
    """

    def __init__(self, n: int = 2, tokenizer=None):
        """
        Initialize the Hindi n-gram tokenizer.

        :param n: Size of n-grams (default: 2 for bigrams)
        :type n: int
        :param tokenizer: Tokenizer to use for initial tokenization.
                         If None, uses HindiWordTokenizer.
        :type tokenizer: IndicTokenizer or None
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self._tokenizer = tokenizer or HindiWordTokenizer()

    def tokenize(self, text: str) -> List[Tuple[str, ...]]:
        """
        Return n-grams from the input text.

        :param text: Input Hindi text
        :type text: str
        :return: List of n-gram tuples
        :rtype: List[Tuple[str, ...]]

        Example:

            >>> tokenizer = HindiNGramTokenizer(n=2)
            >>> tokenizer.tokenize("राम और सीता")
            [('राम', 'और'), ('और', 'सीता')]
        """
        if not text or not text.strip():
            return []

        # First tokenize the text
        tokens = self._tokenizer.tokenize(text)

        if len(tokens) < self.n:
            return []

        # Generate n-grams
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            ngrams.append(ngram)

        return ngrams
