"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # find argmax of prediction
    batch_size, _ = predictions.shape

    predictions_argmax = predictions.argmax(dim=1)

    labels_argmax = labels.argmax(dim=1)
    
    correct = torch.sum(predictions_argmax == labels_argmax)

    accuracy = correct.item() / float(batch_size)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_)
                            for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # neg_slope = FLAGS.neg_slope

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # get the data
    data_dict = cifar10_utils.get_cifar10(
        data_dir=FLAGS.data_dir, validation_size=0)
    trainset = data_dict["train"]
    validationset = data_dict["validation"]
    testset = data_dict["test"]

    # get correct dimensions
    n_channels, height, width = trainset.images[0].shape

    n_input = n_channels * height * width
    n_hidden = dnn_hidden_units
    n_classes = trainset.labels[0].shape[0]
    lr = FLAGS.learning_rate

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # create MLP model
    model = MLP(n_input, n_hidden, n_classes)

    # push model to device
    model.to(device)

    # define loss function
    loss_module = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)

    #######################
    ###### training #######
    #######################

    # set model to train mode
    model.train()

    training_loss = []
    testset_loss = []

    training_accuracy = []
    testset_accuracy = []

    running_loss = 0.0
    running_acc = 0.0

    train_batch_size = FLAGS.batch_size
    eval_frequency = FLAGS.eval_freq


    for iter_step in range(FLAGS.max_steps):
        # get next batch
        train_images, train_labels = trainset.next_batch(train_batch_size)

        # convert numpy array to torch tensor
        train_images = torch.from_numpy(train_images)
        train_labels = torch.from_numpy(train_labels)

        # need to reshape it into vector so batchsize x (3x32x32)
        train_image_batch_vector = torch.reshape(train_images, (train_batch_size, -1))

        # move input data to device if available
        train_image_batch_vector, train_labels = train_image_batch_vector.to(device), train_labels.to(device)

        # run model on input data
        train_output = model(train_image_batch_vector)

        # calculate loss
        # argmax so that loss_module can evaluate output
        train_labels_argmax = torch.argmax(train_labels, dim=1)

        loss = loss_module(train_output, train_labels_argmax)
        
        # perform backpropagation
        optimizer.zero_grad()

        loss.backward()

        # update parameters
        optimizer.step()

        # update running loss
        running_loss += loss.item() 

        # running accuracy
        running_acc += accuracy(train_output, train_labels)

        # if iter_step == eval_frequency:
        if iter_step % eval_frequency == eval_frequency - 1:
            # training loss
            current_loss = running_loss / (eval_frequency)
            training_loss.append(current_loss)
            running_loss = 0.0

            # training acc
            current_acc = running_acc / (eval_frequency)
            training_accuracy.append(current_acc)
            running_acc = 0.0

            # get testset loss and accuracy
            model.eval()

            with torch.no_grad():

                test_epochs_completed = 0.0

                running_test_loss = 0.0
                running_test_acc = 0.0

                test_step_iter = 0.0
                test_batch_size = FLAGS.batch_size

                test_set_img_counter = 0.0
                num_examples_test = testset.num_examples

                while test_set_img_counter < num_examples_test:
                    # get next test batch
                    test_images, test_labels = testset.next_batch(test_batch_size)

                    # convert numpy array to torch tensor
                    test_images = torch.from_numpy(test_images)
                    test_labels = torch.from_numpy(test_labels)

                    # reshape into vector
                    test_images_vector = torch.reshape(test_images, (test_batch_size, -1))

                    # move input data to device if available
                    test_images_vector, test_labels = test_images_vector.to(device), test_labels.to(device)

                    # predictions for test set
                    test_output = model(test_images_vector)

                    test_labels_argmax = torch.argmax(test_labels, dim=1)

                    test_loss = loss_module(test_output, test_labels_argmax)

                    # update running loss
                    running_test_loss += test_loss.item() 

                    # running accuracy
                    running_test_acc += accuracy(test_output, test_labels)

                    test_step_iter += 1

                    test_set_img_counter += test_batch_size

                    
                testset_loss.append(running_test_loss / test_step_iter)
                testset_accuracy.append(running_test_acc / test_step_iter)

                # set model to training mode again
                model.train()

    # print("Final Test Loss")
    # print(testset_accuracy[-1])
    # # plot the train and validation loss
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(len(training_loss)) * eval_frequency + eval_frequency, training_loss, label="Training")
    ax1.plot(np.arange(len(testset_loss)) * eval_frequency + eval_frequency, testset_loss, label="Test")
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training and Test Loss")
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(len(training_accuracy)) * eval_frequency + eval_frequency, training_accuracy, label="Training")
    ax2.plot(np.arange(len(testset_accuracy)) * eval_frequency + eval_frequency, testset_accuracy, label="Test")
    ax2.set_xlabel("Step Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()
    plt.show()
    # fig.savefig("./results/pytorchMLP.png")

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
