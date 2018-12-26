#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################################
# PROGRAMMER: Pierre-Antoine Ksinant                        #
# DATE CREATED: 20/12/2018                                  #
# REVISED DATE: -                                           #
# PURPOSE: Construct a RNN model based on LSTM units        #
#############################################################


##################
# Needed imports #
##################

import nltk, re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


############################
# Function review_to_words #
############################

def review_to_words(review):
    """
    This function transforms a review into its corresponding list of words
    """

    # Download and set removing aspects:
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()

    # Remove HTML tags:
    text = BeautifulSoup(review, "html.parser").get_text()

    # Convert to lower case:
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split string into words:
    words = text.split()

    # Remove stopwords:
    words = [w for w in words if w not in stopwords.words("english")]

    # Remove commoner morphological and inflexional endings:
    words = [PorterStemmer().stem(w) for w in words]

    # Return review as a list of words:
    return words


############################
# Function convert_and_pad #
############################

def convert_and_pad(word_dict, sentence, pad=500):
    """
    This function transforms a sentence through a word dictionary and a padding zone
    """

    # Represent the 'no word' category by '0':
    NOWORD = 0

    # Represent the infrequent words (i.e. not appearing in word_dict) by '1':
    INFREQ = 1

    # Create default working sentence:
    working_sentence = [NOWORD]*pad

    # Treatment of each word of the sentence:
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ

    # Determine the min value between the length of the sentence and the padding zone:
    min_length_zone = min(len(sentence), pad)

    # Return the results:
    return working_sentence, min_length_zone
