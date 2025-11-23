# Natural Language Toolkit: Indic Language Tokenizers
#
# Copyright (C) 2001-2025 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Base classes for Indic language tokenization.

This module provides abstract base classes for tokenizing Indic scripts
(Devanagari, Bengali, Tamil, Telugu, etc.). These classes are designed
to be language-agnostic and extensible for future Indian language support.

The IndicTokenizer base class provides common functionality for all
Indic language tokenizers, while IndicWordTokenizer and IndicSentenceTokenizer
provide specialized interfaces for word and sentence tokenization respectively.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

from nltk.tokenize.api import TokenizerI


class IndicTokenizer(TokenizerI, ABC):
    """
    Base class for Indic language tokenizers.

    This class provides a common interface for tokenizing text in Indic scripts.
    It should NOT contain language-specific logic. Language-specific implementations
    should extend this class or its subclasses.

    Indic scripts share common characteristics:
    - Unicameral alphabet (no uppercase/lowercase)
    - Phonetic ordering
    - Vowel signs (matras) attached to consonants
    - Conjunct consonants
    - Virama (halant) for suppressing inherent vowels

    Subclasses must implement the tokenize() method.
    """

    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets (start_i, end_i),
        where s[start_i:end_i] is the corresponding token.

        :param s: The string to be tokenized
        :type s: str
        :rtype: Iterator[Tuple[int, int]]
        """
        tokens = self.tokenize(s)
        current_pos = 0
        for token in tokens:
            if not token:
                continue
            start = s.find(token, current_pos)
            if start == -1:
                continue
            end = start + len(token)
            current_pos = end
            yield (start, end)


class IndicWordTokenizer(IndicTokenizer, ABC):
    """
    Abstract base class for word tokenization in Indic scripts.

    Word tokenization in Indic scripts must handle:
    - Script-specific word boundaries
    - Punctuation marks (language-specific and universal)
    - Compound words
    - Conjunct consonants (should stay together)
    - Numbers (both script-specific and Arabic numerals)
    - Mixed script text (Indic + Latin for loanwords)

    Subclasses should implement tokenize() with language-specific logic.
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of s, split into words.

        :param s: Input text in Indic script
        :type s: str
        :return: List of word tokens
        :rtype: List[str]
        """
        pass


class IndicSentenceTokenizer(IndicTokenizer, ABC):
    """
    Abstract base class for sentence tokenization in Indic scripts.

    Sentence tokenization in Indic scripts must handle:
    - Script-specific sentence terminators
    - Question marks and exclamation marks
    - Quotation handling
    - Abbreviation detection (to minimize false positives)
    - Verse/section terminators (e.g., double danda in Devanagari)

    Subclasses should implement tokenize() with language-specific logic.
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of s, split into sentences.

        :param s: Input text in Indic script
        :type s: str
        :return: List of sentence strings
        :rtype: List[str]
        """
        pass
