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

import torch.nn as nn


######################################################
# Class to construct a RNN model based on LSTM units #
######################################################

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform sentiment analysis
    """

    # The initialize function:

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by setting up the various layers
        Parameters
         embedding_dim - the size of embeddings
         hidden_dim - the size of the hidden layer outputs
         vocab_size - the number of input dimensions of the neural network (the size of the vocabulary)
        """

        super(LSTMClassifier, self).__init__()

        # Set embedding and layers:
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        # Set class variable:
        self.word_dict = None

    # The forward propagation function:

    def forward(self, x):
        """
        Perform a forward pass of the model on some input
        Parameters
         x - the input to the neural network
        Returns
         y - the output of the neural network
        """

        # Processing of the input:
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]

        # Process forward pass:
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        y = self.sig(out.squeeze())

        # Return the output of the neural network:
        return y
