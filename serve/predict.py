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

import numpy as np
import os, pickle, torch, sagemaker_containers
from model import LSTMClassifier
from utils import review_to_words, convert_and_pad


#####################
# Function model_fn #
#####################

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory
    """

    # Begin loading model:
    print("Loading model: Beginning...\n")

    # First, load the parameters used to create the model:
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    print("*** Model info: {}".format(model_info))

    # Determine the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*** Device: {}".format(device))

    # Construct the model:
    model = LSTMClassifier(model_info['embedding_dim'],
                           model_info['hidden_dim'],
                           model_info['vocab_size'])

    # Load the store model parameters:
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict:
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    # Move to evaluation mode:
    model.to(device).eval()

    # Print built model:
    print("*** Model:\n{}".format(model))

    # End loading model:
    print("\nLoading model: Done...")

    # Return model:
    return model


#####################
# Function input_fn #
#####################

def input_fn(serialized_input_data, content_type):
    """
    Deserialize the input data
    """

    # Begin deserializing:
    print("Deserializing the input data...")

    # Check content type:
    if content_type == "text/plain":
        data = serialized_input_data.decode('utf-8')
        print("Done.")
        # Return deserialized data:
        return data

    # Content type not supported:
    raise Exception("Requested unsupported ContentType in content_type: {}".format(content_type))


######################
# Function output_fn #
######################

def output_fn(prediction_output, accept):
    """
    Serialize the output data
    """

    # Begin serializing:
    print("Serializing the generated output...")

    # Perform serializing:
    serialized_prediction_output = str(prediction_output)
    print("Done.")

    # Return serialized prediction output:
    return serialized_prediction_output


#######################
# Function predict_fn #
#######################

def predict_fn(input_data, model):
    """
    Make prediction on input data thanks to model
    """

    # Begin predicting:
    print("Inferring sentiment of input data...")

    # Determine the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check model word dictionary presence:
    if model.word_dict is None:
        raise Exception("Model has not been loaded properly: no word_dict")

    # Process input_data so that it is ready to be sent to our model:
    data_x, data_len = convert_and_pad(model.word_dict, review_to_words(input_data))

    # Construct an appropriate input tensor:
    data_pack = np.hstack((data_len, data_x))
    data_pack = data_pack.reshape(1, -1)
    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode:
    model.eval()

    # Apply model to input tensor:
    output_data = model.forward(data)

    # Transform output into a numpy array which contains a single integer, 1 or 0:
    if torch.cuda.is_available():
        # NumPy doesn't support CUDA:
        output_data.to('cpu')
    result = int(np.round(output_data.detach().numpy()))

    # Return prediction:
    return result
