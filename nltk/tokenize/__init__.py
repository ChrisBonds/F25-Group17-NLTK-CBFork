# Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# Contributors: matthewmc, clouds56
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

r"""
NLTK Tokenizer Package

Tokenizers divide strings into lists of substrings.  For example,
tokenizers can be used to find the words and punctuation in a string:

    >>> from nltk.tokenize import word_tokenize
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
    ... two of them.\n\nThanks.'''
    >>> word_tokenize(s) # doctest: +NORMALIZE_WHITESPACE
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.',
    'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

This particular tokenizer requires the Punkt sentence tokenization
models to be installed. NLTK also provides a simpler,
regular-expression based tokenizer, which splits text on whitespace
and punctuation:

    >>> from nltk.tokenize import wordpunct_tokenize
    >>> wordpunct_tokenize(s) # doctest: +NORMALIZE_WHITESPACE
    ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.',
    'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

We can also operate at the level of sentences, using the sentence
tokenizer directly as follows:

    >>> from nltk.tokenize import sent_tokenize, word_tokenize
    >>> sent_tokenize(s)
    ['Good muffins cost $3.88\nin New York.', 'Please buy me\ntwo of them.', 'Thanks.']
    >>> [word_tokenize(t) for t in sent_tokenize(s)] # doctest: +NORMALIZE_WHITESPACE
    [['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.'],
    ['Please', 'buy', 'me', 'two', 'of', 'them', '.'], ['Thanks', '.']]

Caution: when tokenizing a Unicode string, make sure you are not
using an encoded version of the string (it may be necessary to
decode it first, e.g. with ``s.decode("utf8")``.

NLTK tokenizers can produce token-spans, represented as tuples of integers
having the same semantics as string slices, to support efficient comparison
of tokenizers.  (These methods are implemented as generators.)

    >>> from nltk.tokenize import WhitespaceTokenizer
    >>> list(WhitespaceTokenizer().span_tokenize(s)) # doctest: +NORMALIZE_WHITESPACE
    [(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36), (38, 44),
    (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]

There are numerous ways to tokenize text.  If you need more control over
tokenization, see the other methods provided in this package.

For further information, please see Chapter 3 of the NLTK book.
"""

import functools
import re
from typing import List, Tuple

from nltk.data import load
from nltk.tokenize.casual import TweetTokenizer, casual_tokenize
from nltk.tokenize.destructive import NLTKWordTokenizer

# Hindi tokenizers
from nltk.tokenize.hindi import (
    HindiCharacterTokenizer,
    HindiNGramTokenizer,
    HindiSentenceTokenizer,
    HindiSubwordTokenizer,
    HindiWordTokenizer,
)
from nltk.tokenize.legality_principle import LegalitySyllableTokenizer
from nltk.tokenize.mwe import MWETokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTokenizer
from nltk.tokenize.regexp import (
    BlanklineTokenizer,
    RegexpTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
    blankline_tokenize,
    regexp_tokenize,
    wordpunct_tokenize,
)
from nltk.tokenize.repp import ReppTokenizer
from nltk.tokenize.sexpr import SExprTokenizer, sexpr_tokenize
from nltk.tokenize.simple import (
    LineTokenizer,
    SpaceTokenizer,
    TabTokenizer,
    line_tokenize,
)
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize


@functools.lru_cache
def _get_punkt_tokenizer(language="english"):
    """
    A constructor for the PunktTokenizer that utilizes
    a lru cache for performance.

    :param language: the model name in the Punkt corpus
    :type language: str
    """
    return PunktTokenizer(language)


# Standard sentence tokenizer.
def sent_tokenize(text, language="english"):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = _get_punkt_tokenizer(language)
    return tokenizer.tokenize(text)


# Standard word tokenizer.
_treebank_word_tokenizer = NLTKWordTokenizer()


def word_tokenize(text, language="english", preserve_line=False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :type text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: A flag to decide whether to sentence tokenize the text or not.
    :type preserve_line: bool
    """
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [
        token for sent in sentences for token in _treebank_word_tokenizer.tokenize(sent)
    ]


# Hindi tokenization convenience functions
_hindi_word_tokenizer = HindiWordTokenizer()
_hindi_sentence_tokenizer = HindiSentenceTokenizer()
_hindi_character_tokenizer = HindiCharacterTokenizer()


def hindi_word_tokenize(text: str, language: str = "hindi") -> List[str]:
    """
    Tokenize Hindi text into words.

    :param text: Input Hindi text in Devanagari script
    :type text: str
    :param language: Language identifier (default: 'hindi')
    :type language: str
    :return: List of word tokens
    :rtype: List[str]

    Example:

        >>> from nltk.tokenize import hindi_word_tokenize
        >>> text = "राम और सीता वन में गए।"
        >>> hindi_word_tokenize(text)
        ['राम', 'और', 'सीता', 'वन', 'में', 'गए', '।']
    """
    return _hindi_word_tokenizer.tokenize(text)


def hindi_sent_tokenize(text: str, language: str = "hindi") -> List[str]:
    """
    Tokenize Hindi text into sentences.

    :param text: Input Hindi text in Devanagari script
    :type text: str
    :param language: Language identifier (default: 'hindi')
    :type language: str
    :return: List of sentence strings
    :rtype: List[str]

    Example:

        >>> from nltk.tokenize import hindi_sent_tokenize
        >>> text = "राम वन गया। सीता घर रही।"
        >>> hindi_sent_tokenize(text)
        ['राम वन गया।', 'सीता घर रही।']
    """
    return _hindi_sentence_tokenizer.tokenize(text)


def hindi_char_tokenize(text: str) -> List[str]:
    """
    Character-level tokenization for Hindi text.

    Handles Unicode Devanagari character boundaries, keeping vowel signs
    (matras) with their base consonants and preserving conjunct consonants.

    :param text: Input Hindi text
    :type text: str
    :return: List of character tokens
    :rtype: List[str]

    Example:

        >>> from nltk.tokenize import hindi_char_tokenize
        >>> hindi_char_tokenize("राम")
        ['रा', 'म']
    """
    return _hindi_character_tokenizer.tokenize(text)


def hindi_ngrams(text: str, n: int = 2) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from Hindi text.

    :param text: Input Hindi text
    :type text: str
    :param n: Size of n-grams (default: 2 for bigrams)
    :type n: int
    :return: List of n-gram tuples
    :rtype: List[Tuple[str, ...]]

    Example:

        >>> from nltk.tokenize import hindi_ngrams
        >>> hindi_ngrams("राम और सीता", n=2)
        [('राम', 'और'), ('और', 'सीता')]
    """
    tokenizer = HindiNGramTokenizer(n=n)
    return tokenizer.tokenize(text)
