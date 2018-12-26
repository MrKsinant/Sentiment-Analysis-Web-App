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

import pandas as pd
import argparse, json, os, pickle, torch, time, sagemaker_containers
from model import LSTMClassifier
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


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


###################################
# Function _get_train_data_loader #
###################################

def _get_train_data_loader(batch_size, training_dir):
    """
    Create data loader from train data file
    """

    # Begin construction:
    print("Get train data loader...")

    # Read train data file:
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Create data loader:
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()
    train_ds = TensorDataset(train_X, train_y)
    data_loader = DataLoader(train_ds, batch_size=batch_size)

    # End construction:
    print("Done.")

    # Return data loader:
    return data_loader


##################
# Function train #
##################

def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training function that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    # Track the training session:
    print("Training (for {} epoch(s)):\n*****".format(epochs))
    start_time = time.time()

    # Perform forward and backpropagation passes:
    for epoch in range(1, epochs + 1):

        # Put model in training mode:
        model.train()

        # Set total loss to zero:
        total_loss = 0

        # Move on training data loader:
        for batch in train_loader:

            # Define data and label:
            batch_X, batch_y = batch

            # Move to consistent device:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero accumulated gradients:
            optimizer.zero_grad()

            # Get the output from the model:
            batch_output = model.forward(batch_X)

            # Calculate the loss:
            loss = loss_fn(batch_output, batch_y)

            # Perform backpropagation:
            loss.backward()

            # Perform optimization:
            optimizer.step()

            # Calculate loss over data loader:
            total_loss += loss.item()

        # Print and register loss stats:
        print("Epoch {}... Loss {}...".format(epoch, total_loss/len(train_loader)))

    # Time performance:
    end_time = time.time()
    total_time = int(end_time - start_time)
    hours = total_time//3600
    minutes = (total_time%3600)//60
    seconds = (total_time%3600)%60
    print("*****\nEnd of the training: {:02d}h {:02d}m {:02d}s".format(hours,
                                                                       minutes,
                                                                       seconds))


######################################################################################
# Send model parameters and training parameters as arguments through argument parser #
######################################################################################

if __name__ == '__main__':

    # Create parser:
    parser = argparse.ArgumentParser()

    # Training parameters:
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model parameters:
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker parameters:
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # Create namespace for parser:
    args = parser.parse_args()

    # Determine the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}...".format(device))

    # Set the seed:
    torch.manual_seed(args.seed)

    # Load the training data:
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model:
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    # Add word dictionary to model:
    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    # Print model characteristics:
    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}...".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model:
    optimizer = Adam(model.parameters())
    loss_fn = BCELoss()
    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model:
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

	# Save the word_dict:
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters:
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
