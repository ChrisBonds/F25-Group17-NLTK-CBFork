# Natural Language Toolkit: Stemmers
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
#         Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Basic Hindi Stemmer

This is a basic stemmer for Hindi language that removes common suffixes of 4 categories of lengths.
These categories were primarily created on visual length upon viewing the suffix,
BUT they have different logical lengths in python itself. 
This is why the categories do not directly reflect their character length in python.
Solving this issue was what helped pass all the 40 test cases in the report. 

The lengths are categorized as follows:
Category 1: All suffixes are of length 1 in python.
Category 2: Different classification of symbols, all suffixes are of length 1 in python.
Category 3: Some suffixes are length 2 in python, some suffixes are length 3 in python.
Category 4, All suffixes are of length 4 in python.
BUT, sometimes removing a length 4 suffix may require adding back a character to RETRIEVE THE STEM.

"""

from nltk.stem.api import StemmerI


class HindiStemmer(StemmerI):
    """
    Hindi stemmer based on suffix removal algorithm.
    
    This stemmer removes common Hindi suffixes to extract the root meaning
    of words. It processes suffixes in order of length (longest first) to
    handle complex morphological structures.
    
    >>> from nltk.stem import HindiStemmer
    >>> stemmer = HindiStemmer()
    >>> stemmer.stem('किताबों')
    'किताब'
    >>> stemmer.stem('लड़कियों')
    'लड़की'
    >>> stemmer.stem('खाता')
    'खा'
    """

    def __init__(self):
        """
        Initialize the Hindi stemmer with suffix lists.
        
        Suffixes are organized by length, with longer suffixes checked first
        to handle complex morphological structures correctly.
        """
        # Length 4 suffixes (longest first)
        self.suffixes_4 = [
            'करके',  # after doing
            'करना',  # to do
            'करने',  # doing (oblique)
            'करनी',  # to do (feminine)
            'करते',  # doing (plural)
            'करती',  # doing (feminine)
            'करता',  # doing (masculine)
            'होने',  # being (oblique)
            'होना',  # to be
            'होती',  # being (feminine)
            'होता',  # being (masculine)
            'रही',   # continuous (feminine)
            'रहे',   # continuous (plural)
            'रहा',   # continuous (masculine)
            'ियों',  # plural marker (feminine oblique) - इ + य + ो + ं
        ]
        
        # Length 3 suffixes
        self.suffixes_3 = [
            'कर',    # do/make
            'हो',    # be
            'से',    # from/with (case marker)
            'में',   # in (case marker)
            'पर',    # on (case marker)
            'तक',    # until (case marker)
            'को',    # to (case marker)
            'का',    # of (masculine, case marker)
            'की',    # of (feminine, case marker)
            'के',    # of (plural, case marker)
            'ने',    # by (case marker, ergative)
            'तो',    # then/if
            'भी',    # also
            'ही',    # only/emphasis
            'गा',    # will (masculine)
            'गी',    # will (feminine)
            'गे',    # will (plural)
            'ता',    # present tense (masculine)
            'ती',    # present tense (feminine)
            'ते',    # present tense (plural)
            'या',    # past tense (masculine)
            'यी',    # past tense (feminine)
            'ये',    # past tense (plural)
            'एं',    # plural marker
            'ओं',    # plural marker (oblique) - ओ + ं
            'ों',     # plural marker (oblique) - ो + ं (alternative form)
            'यों',   # plural marker
        ]
        
        # Length 2 suffixes
        self.suffixes_2 = [
            'ए',     # plural/vocative
            'ओ',     # vocative/plural
            'आ',     # masculine marker
            'ई',     # feminine marker
            'उ',     # some verb forms
            'ऊ',     # some verb forms
        ]
        
        # Length 1 suffixes (checked last)
        self.suffixes_1 = [
            'ा',     # long a (masculine)
            'ी',     # long i (feminine)
            'े',     # long e (plural/oblique)
            'ो',     # long o (oblique)
            'ु',     # short u
            'ू',     # long u
        ]

    def stem(self, token):
        """
        Strip affixes from the token and return the stem.
        
        This method removes suffixes in order of length (longest first)
        to handle complex morphological structures. It ensures that
        the resulting stem has a minimum length to avoid over-stemming.
        
        :param token: The token that should be stemmed.
        :type token: str
        :return: The stemmed token.
        :rtype: str
        """
        if not token:
            return token
        
        word = token
        original_length = len(word)
        
        # Minimum stem length to prevent over-stemming
        min_stem_length = 2
        
        # Combine all suffixes and sort by length (longest first)
        # This ensures longer suffixes are checked before shorter ones
        all_suffixes = []
        for suffix in self.suffixes_4:
            all_suffixes.append((len(suffix), suffix))
        for suffix in self.suffixes_3:
            all_suffixes.append((len(suffix), suffix))
        for suffix in self.suffixes_2:
            all_suffixes.append((len(suffix), suffix))
        for suffix in self.suffixes_1:
            all_suffixes.append((len(suffix), suffix))
        
        # Sort by length descending, then by suffix string to ensure consistent ordering
        all_suffixes.sort(key=lambda x: (-x[0], x[1]))
        
        # Try to remove suffixes in order of length (longest first)
        for suffix_len, suffix in all_suffixes:
            # Check if word is long enough to have this suffix removed
            # We need: original_length >= min_stem_length + suffix_len
            if original_length >= min_stem_length + suffix_len:
                if word.endswith(suffix):
                    stem = word[:-suffix_len]
                    if len(stem) >= min_stem_length:
                        # Special morphological cases for Hindi
                        # When removing 'यों', if stem ends with 'ि', change to 'ी'
                        # This handles cases like 'लड़कियों' -> 'लड़की' (not 'लड़कि')
                        # if suffix == 'यों' and stem.endswith('ि'):
                        #     # stem = stem[:-1] + 'ी'
                        # When removing 'ियों', add 'ी' to get the correct feminine form
                        # This handles 'लड़कियों' -> 'लड़की'
                        if suffix == 'ियों':
                            stem = stem + 'ी'
                        return stem
        
        # If no suffix was removed, return the original word
        return word