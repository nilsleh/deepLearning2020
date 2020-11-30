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
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

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

def make_char_prediction(config, gen_output):

    if config.sample_method == "greedy":
        softmax_pred = torch.nn.functional.log_softmax(gen_output[-1, :, :], dim=1)
        
        predictions = torch.argmax(softmax_pred, dim=1)
        # gen_text[t+len(numeric_rep), :] = predictions.type(torch.LongTensor)
        return predictions.type(torch.LongTensor)

    elif config.sample_method == "temp":
        temp_output = gen_output * config.temperature
        softmax_pred = torch.nn.functional.softmax(temp_output[-1, :, :], dim=1)
    
        sample = torch.multinomial(softmax_pred, 1)
        # gen_text[t+len(numeric_rep), :] = samples.permute(1,0).type(torch.LongTensor)

        return sample.permute(1,0).type(torch.LongTensor)


def finish_sentence(config, dataset, dataloader, vocab_size):
    # load model
    device = torch.device(config.device)
    print("Finish sentence")
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden
        , config.lstm_num_layers, device)
    model.load_state_dict(torch.load(config.path_saved_model, map_location=lambda storage, loc: storage))
    model.eval()

    if config.finish_sentence == " ":
        num_examples = 5
        # generate sequences from scratch
        gen_text = torch.zeros(config.sample_length, num_examples).type(torch.LongTensor)
        gen_text = gen_text.to(device)
        gen_text[0, :] = torch.tensor(random.sample(range(vocab_size), num_examples)).type(torch.LongTensor)

        # generate initial zero states
        h_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden)).to(device)
        c_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden)).to(device)

        # input sentence len is zero 
        input_sentence_len = 0
        
        # how many chars to fill
        length_to_fill = gen_text.size(0)

    else:
        # finish the given input sentence
        # turn string into digit representation
        numeric_rep = dataset.convert_to_number(config.finish_sentence)
        
        input_sentence_len = len(numeric_rep)
        num_examples = 1
        
        gen_text = torch.zeros(config.sample_length * 3, num_examples).type(torch.LongTensor)

        numeric_rep = torch.tensor(numeric_rep).unsqueeze(1).type(torch.LongTensor)
        numeric_reps = numeric_rep.repeat(1, num_examples)

        # insert numeric_reps to gen_text
        gen_text[0:input_sentence_len, :] = numeric_reps

        h_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden))
        c_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden))
        
        length_to_fill = gen_text.size(0) - input_sentence_len

        # put sentence that you want to finish through model
        gen_output, h_state, c_state = model(gen_text[0:input_sentence_len,:], h_state, c_state)

        gen_text[input_sentence_len, :] = make_char_prediction(config, gen_output)
    
    # subsequently predict the next characters up to wished length
    for t in range(length_to_fill-1):
        model_input = gen_text[t+input_sentence_len,:][None, :]
        gen_output, h_state, c_state = model(model_input, h_state, c_state)
        pred = make_char_prediction(config, gen_output)
        
        gen_text[t+input_sentence_len+1, :] = pred
 
    # turn digits into characters
    print("{:s} :".format(config.finish_sentence))
    for exm in range(num_examples):
        print("Example {:d}".format(exm))
        example = gen_text[:, exm].tolist()
        char_example = dataset.convert_to_string(example)
        print(char_example)


def load_dataset(config):
    dataset = TextDataset(config.txt_file, config.seq_length)  # FIXME
    data_loader = DataLoader(dataset, config.batch_size)
    vocab_size = dataset.vocab_size

    return dataset, data_loader, vocab_size
                    

def train(config, dataset, dataloader, vocab_size):

    print("Vocabulary Size {:d}".format(vocab_size))
    tb = SummaryWriter()

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden
        , config.lstm_num_layers, device, config.dropout_keep_prob).to(device)  # FIXME

    # Setup the loss and optimizer
    loss_module = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # FIXME

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.learning_rate_step)

    train_step = 0

    losses = []

    accuracies = []

    best_patience_acc = -1

    steps = []

    while train_step < config.train_steps:

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            steps.append(step)
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

            # version to pass it directly to loss_module
            # output reshape for Cross entropy loss [batch_size, vocab_size, seq_length]
            # output_reshape = output.permute(1, 2, 0)
            # targets_reshape = batch_targets.permute(1, 0)

            # loss = loss_module(output_reshape, targets_reshape)

            # accuracy = calculate_accuracy(output, batch_targets)

            # version with looping over sequence length
            seq_losses = 0
            seq_correct = 0
            for seq_step in range(config.seq_length):
                # loss
                seq_output = output[seq_step, :, :]
                seq_targets = batch_targets[seq_step, :]
                seq_loss = loss_module(seq_output, seq_targets)
                seq_losses += seq_loss

                # accuracy
                predictions = torch.argmax(seq_output, dim=1)
                correct = (predictions == seq_targets).sum().item()
                seq_correct += correct 

            # loss
            loss = seq_losses / config.batch_size
            accuracy = seq_correct / (config.batch_size * config.seq_length)
            loss_val = loss.item()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)

            # update parameters
            optimizer.step()

            # lr_scheduler step
            scheduler.step()

            # update training_steps
            train_step += 1

            # append loss and accuracy
            # losses.append(loss.item)
            losses.append(loss_val)
            accuracies.append(accuracy)

            
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
                tb.add_scalar("Loss", loss, train_step)
                tb.add_scalar("Accuracy", accuracy, train_step)
                tb.add_scalar("Learning Rate", scheduler.get_last_lr()[0], train_step)

            # check patience for early stopping
            if config.early_stopping:
                if len(accuracies) == 1 or accuracy > accuracies[best_patience_acc]:
                    best_patience_acc = train_step
                elif best_patience_acc <= (train_step + 1) - config.patience:
                    print("Early stopping invoked at step {:d} because accuracy has not improved over last {:d}".format(
                        train_step, config.patience))
                    print("Last reported:")
                    print("Accuracy = {:.2f}, Loss = {:.3f}".format(accuracy, loss))
                    break

            # Generate some sentences by sampling from the model
            if (train_step) % int(config.train_steps / 3) == 0:
                # set model to eval
                model.eval()

                num_examples = 5
                
                gen_text = torch.zeros(config.sample_length, num_examples).type(torch.LongTensor)
                gen_text = gen_text.to(device)

                gen_text[0, :] = torch.tensor(random.sample(range(vocab_size), num_examples)).type(torch.LongTensor)

                # generate initial zero states
                h_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden)).to(device)
                c_state = torch.zeros((config.lstm_num_layers, num_examples, config.lstm_num_hidden)).to(device)
                
                for t in range(config.sample_length-1):
                    gen_output, h_state, c_state = model(gen_text[t, :][None,:], h_state, c_state)
                    gen_text[t+1, :] = make_char_prediction(config, gen_output)

                # turn digits into characters
                print("Generated Text at trainstep {:d}, accuracy{:f}".format(train_step, accuracy))
                for exm in range(num_examples):
                    example = gen_text[:, exm].tolist()
                    char_example = dataset.convert_to_string(example)
                    final = char_example[:config.seq_length] + 'II' + char_example[config.seq_length:]
                    print(final)
                

                # set model to train again
                model.train()

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
    
    print('Done training.')
    # save model
    if config.save_model:
        torch.save(model.state_dict(), config.path_saved_model)
        print("Model saved")



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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
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
    parser.add_argument('--early_stopping', type=bool, default=False,
                        help="Whether or not to invoke early stopping")
    parser.add_argument('--patience', type=int, default=2000,
                        help="Early stopping number of training steps")

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100,
                        help='How often to print training progress')
    parser.add_argument('--sample_method', type=str, default="greedy",
                        help='Whether to generate new text by greedy or random sampling: greedy or random')
    parser.add_argument('--sample_length', type=int, default=30,
                        help="How many characters the sampled sentence should have")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature parameter for the softmax function")
    parser.add_argument('--finish_sentence', type=str, default=" ",
                        help="Type a sentence that you would like the model to finish.\
                            With the default value, the model will generate random chars from scratch.")
    parser.add_argument('--train_or_finish', type=str, default="train", 
                        help="Do only training, or only finish sentences or both")
    parser.add_argument('--save_model', type=bool, default=False,
                        help="Decide if you want to save the model or not.")
    parser.add_argument('--path_saved_model', type=str, default="lstm_grimms.pt",
                        help="Path to saved model")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # load datasets
    dataset, data_loader, vocab_size = load_dataset(config)

    # define a seed
    torch.manual_seed(0)

    if config.train_or_finish == 'train':
        # added print flags
        print_flags()
        train(config, dataset, data_loader, vocab_size)

    elif config.train_or_finish == 'finish':
        finish_sentence(config, dataset, data_loader, vocab_size)
    
    elif config.train_or_finish == "both":
        # added print flags
        print_flags()
        train(config, dataset, data_loader, vocab_size)       
        finish_sentence(config, dataset, data_loader, vocab_size)