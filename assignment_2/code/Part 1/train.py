###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM

import numpy as np
import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
from torch.utils.tensorboard import SummaryWriter

###############################################################################


def plot_results(train_losses, train_accuracies, test_losses, test_accuracies):
    '''
    Plots training losses and accuracy as well as a print statement about loss
    and accuracy on testset
    '''
    ##### training results #####
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)

    # loop over different sequence lengths
    for seq_len, values in train_losses.items():
        loss_array = turn_var_size_list_to_np(values)
        steps = np.arange(loss_array.shape[1])
        mean_loss = np.mean(loss_array, axis=0)
        ax1.plot(steps, mean_loss, label="T=" + str(seq_len))
    
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Loss for different sequence lengths T")
    ax1.legend(loc="upper right")
    ax1.set_xlim(steps[0], steps[-1])

    ax2 = fig.add_subplot(122)

    # loop over different sequence lengths
    for seq_len, values in train_accuracies.items():
        acc_array = turn_var_size_list_to_np(values)
        steps = np.arange(acc_array.shape[1])
        mean_acc = np.mean(acc_array, axis=0)
        ax2.plot(steps, mean_acc, label="T=" + str(seq_len))

    ax2.set_xlabel("Step Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy for different sequence lengths T")
    ax2.legend(loc="lower right")
    ax2.set_xlim(steps[0], steps[-1])

    plt.show()
    fig.savefig("./results/" + str(config.model_type) + ".png")

    ##### test results #####
    print("Test Results Losses")
    first_key = list(test_losses.keys())[0]
    test_loss_results = np.zeros(
        (len(test_losses), len(test_losses[first_key])))
    i = 0
    for seq_len, values in test_losses.items():
        loss_array = np.array(values)
        test_loss_results[i, :] = loss_array
        print("Sequnce Length {:d}, Mean: {:.4f}, std: {:.4f}".format(
            seq_len, np.mean(loss_array), np.std(loss_array)))
        i += 1

    print("Test Results Accuracies")
    first_key = list(test_accuracies.keys())[0]
    test_acc_results = np.zeros(
        (len(test_accuracies), len(test_accuracies[first_key])))
    i = 0
    for seq_len, values in test_accuracies.items():
        acc_array = np.array(values)
        test_acc_results[i, :] = acc_array
        print("Sequnce Length {:d}, Mean: {:.4f}, std: {:.4f}".format(
            seq_len, np.mean(acc_array), np.std(acc_array)))
        i += 1


def turn_var_size_list_to_np(values):
    '''
    Helper function, that returns a numpy array from a list of list with varying size that
    can be encountered when applying early stopping
    '''
    copy_values = values.copy()

    row_num = []
    for row in copy_values:
        row_num.append(len(row))

    max_length = max(row_num)

    for row in copy_values:
        while len(row) < max_length:
            row.append(np.nan)

    values_array = np.array(copy_values)

    return values_array


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = config.input_length*4+2-1

    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device, config.embed_size
        ).to(device)

    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []

    best_patience_acc = -1

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        output = model(batch_inputs)

        # Compute the loss, gradients and update network parameters
        loss = loss_function(output, batch_targets.type(torch.LongTensor))

        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        predictions = torch.argmax(output, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / output.size(0)

        if accuracy > 1.0:
            print("Accuracy should not be larger than 1")
            

        # track losses and accuracies
        losses.append(loss.item())
        accuracies.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.track_freq == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                      datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                      config.train_steps, config.batch_size, examples_per_second,
                      accuracy, loss
                  ))

        # check patience for early stopping
        if config.early_stop:
        
            if len(accuracies) == 1 or accuracy > accuracies[best_patience_acc]:
                best_patience_acc = step
            elif best_patience_acc <= (step + 1) - config.patience:
                print("Early stopping invoked at step {:d} because accuracy has not improved over last {:d}".format(
                    step, config.patience))
                print("Last reported:")
                print("Accuracy = {:.2f}, Loss = {:.3f}".format(accuracy, loss))
                break

        # Check if training is finished
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    # final evaluation after being done with training on a testset
    test_set_batch_size = 100
    num_test_steps = 10
    testset_loader = DataLoader(dataset, test_set_batch_size, num_workers=0,  # change num workers to 0
                                drop_last=True)

    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    run_acc_ls = []
    for test_step, (test_inputs, test_targets) in enumerate(testset_loader):
        # Move to GPU
        test_inputs = test_inputs.to(device)     # [batch_size, seq_length,1]
        test_targets = test_targets.to(device)

        test_output = model(test_inputs)

        test_loss = loss_function(
            test_output, test_targets.type(torch.LongTensor))

        running_loss += test_loss.item()

        test_preds = torch.argmax(test_output, dim=1)
        test_correct = (test_preds == test_targets).sum().item()

        running_acc += test_correct / test_output.size(0)
        run_acc_ls.append(test_correct / test_output.size(0))

        if test_step == (num_test_steps-1):
            break

    final_test_loss = running_loss / num_test_steps
    final_test_acc = running_acc / num_test_steps

    return losses, accuracies, final_test_loss, final_test_acc

    ###########################################################################
    ###########################################################################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='randomcomb',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    parser.add_argument('--model_type', type=str, default='biLSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=6,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--embed_size', type=int, default=1,
                        help='Dimension size of the word embedding')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--patience', type=int, default=500,
                        help="Early stopping number of training steps")
    parser.add_argument('--early_stop', type=bool, default=True,
                        help="Whether or not to apply early stopping.")

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--track_freq', type=int, default=100,
                        help="How frequently to track loss and accuracy")

    config = parser.parse_args()

    print_flags()

    # define three different seeds
    seeds = [0, 24, 42]
    seq_lengths = [5, 10, 15]
    train_losses = {}
    train_accuracies = {}

    test_losses = {}
    test_accuracies = {}

    for seq in seq_lengths:
        print("Training Sequence {:d}".format(seq))
        config.input_length = seq
        train_losses[seq] = []
        train_accuracies[seq] = []

        test_losses[seq] = []
        test_accuracies[seq] = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            train_loss, train_acc, test_loss, test_acc = train(config)

            train_losses[seq].append(train_loss)
            train_accuracies[seq].append(train_acc)

            test_losses[seq].append(test_loss)
            test_accuracies[seq].append(test_acc)

    # plot the results
    plot_results(train_losses, train_accuracies, test_losses, test_accuracies)
