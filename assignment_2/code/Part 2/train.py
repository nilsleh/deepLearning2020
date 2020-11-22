# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from part2.dataset import TextDataset
# from part2.model import TextGenerationModel

from dataset import TextDataset
from model import TextGenerationModel

import random

# from torch.utils.tensorboard import SummaryWriter

###############################################################################
def calculate_accuracy(output, targets):
    correct = 0
    for seq_step in range(config.seq_length):
        seq_output = output[seq_step, :, :]
        seq_targets = targets[seq_step, :]
        # accuracy
        predictions = torch.argmax(seq_output, dim=1)
        seq_correct = (predictions == seq_targets).sum().item()
        correct += seq_correct 

    accuracy = correct / (config.batch_size * config.seq_length)

    return accuracy

def train(config):


    # tb = SummaryWriter()

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # FIXME
    data_loader = DataLoader(dataset, config.batch_size)

    # inspect data 
    # batch_in, batch_tar = next(iter(data_loader))
    vocab_size = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden
        , config.lstm_num_layers, device).to(device)  # FIXME

    # Setup the loss and optimizer
    # criterion average of cross-entropy losses
    # loss_module = nn.NLLLoss()  # FIXME
    loss_module = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # FIXME

    train_step = 0

    while train_step < config.train_steps:

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            # turn data into tensors
            batch_inputs = torch.stack(batch_inputs)
            batch_targets = torch.stack(batch_targets)

            # Move to GPU
            batch_inputs = batch_inputs.to(device)    
            batch_targets = batch_targets.to(device)

            # Reset for next iteration
            model.zero_grad()

            # forward pass
            # output shape [seq_length, batch_size, vocab_size]
            output = model(batch_inputs)

            # output reshape for Cross entropy loss [batch_size, vocab_size, seq_length]
            output_reshape = output.permute(1, 2, 0)
            targets_reshape = batch_targets.permute(1, 0)

            loss = loss_module(output_reshape, targets_reshape)

            accuracy = calculate_accuracy(output, batch_targets)

            # seq_losses = 0
            # seq_correct = 0
            # for seq_step in range(config.seq_length):
            #     # loss
            #     seq_output = output[seq_step, :, :]
            #     seq_targets = batch_targets[seq_step, :]
            #     seq_loss = loss_module(seq_output, seq_targets)
            #     seq_losses += seq_loss

            #     # accuracy
            #     predictions = torch.argmax(seq_output, dim=1)
            #     correct = (predictions == seq_targets).sum().item()
            #     seq_correct += correct    
            # # loss
            # loss = seq_losses / config.batch_size
            # accuracy = seq_correct / (config.batch_size * config.seq_length)
            # loss_val = loss.item()

            loss.backward()

            # loss = np.inf   # FIXME
            # accuracy = 0.0  # FIXME

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)

            # update parameters
            optimizer.step()


            # update training_steps
            train_step += 1

            
            #######################################################

        
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if (train_step) % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                        ))

                # log results
                # tb.add_scalar("Loss", loss, train_step)
                # tb.add_scalar("Accuracy", accuracy, train_step)

            # Generate some sentences by sampling from the model
            if (train_step) % config.sample_every == 0:
                # set model to eval
                model.eval()

                if config.sample_method == "greedy":
                    num_examples = 5
                    # create tensor that holds result and initialiize first character
                    gen_text = torch.zeros(config.sample_length, num_examples).type(torch.LongTensor)
                    gen_text[0, :] = torch.tensor(random.sample(range(vocab_size), num_examples)).type(torch.LongTensor)

                    # tensor to hold the results
                    # sentences = torch.empty(config.sample_length, 5)
                    
                    for t in range(config.sample_length-1):

                        # output shape [t, num_examples, vocab_size]
                        sample_output = model(gen_text[0:t+1, :])

                        # only interested in latest prediction
                        new_char = sample_output[-1, :, :]

                        predictions = torch.argmax(new_char, dim=1)
                        
                        gen_text[t+1, :] = predictions.type(torch.LongTensor)

                    # turn digits into characters
                    print("Generated Text at trainstep %d".format(train_step))
                    for exm in range(num_examples):
                        example = gen_text[:, exm].tolist()
                        char_example = dataset.convert_to_string(example)
                        print(char_example)

                    
                # elif config.sampling_method == "random":


                # set model to train again
                model.train()

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--sample_method', type=str, default="greedy",
                        help='Whether to generate new text by greedy or random sampling: greedy or random')
    parser.add_argument('--sample_length', type=int, default=30,
                        help="How many characters the sampled sentence should have")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # added print flags
    print_flags()

    # Train the model
    train(config)
