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
    batch_size, n_classes = predictions.shape

    predictions_argmax = predictions.argmax(dim=1)

    prediction_matrix = F.one_hot(predictions_argmax, num_classes=n_classes)

    result = prediction_matrix * labels

    correct = torch.sum(torch.sum(result, dim=1))

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

    # TODO Something that tracks runtime

    # get the data
    # TODO get_cifar10 takes parameter that specifies size of validation set
    data_dict = cifar10_utils.get_cifar10(
        data_dir=FLAGS.data_dir, validation_size=0)
    trainset = data_dict["train"]
    validationset = data_dict["validation"]
    testset = data_dict["test"]

    n_input = 3 * 32 * 32
    n_hidden = [100]
    n_classes = 10

    # create MLP model
    model = MLP(n_input, n_hidden, n_classes)

    # define loss function
    loss_module = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.learning_rate)

    # num examples train
    num_examples_train = trainset.num_examples
    print(num_examples_train)

    # num examples validation
    num_examples_valid = validationset.num_examples
    print(num_examples_valid)

    # num examples test
    num_examples_test = testset.num_examples
    print(num_examples_test)

    #######################
    ###### training #######
    #######################

    # set model to train mode
    model.train()

    eval_freq_counter = 0

    training_loss = []
    testset_loss = []

    training_accuracy = []
    testset_accuracy = []

    for iter_step in range(FLAGS.max_steps):

        # get next batch
        images, labels = trainset.next_batch(FLAGS.batch_size)

        # convert numpy array to torch tensor
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        # need to reshape it into vector so batchsize x (3x32x32)
        image_batch_vector = torch.reshape(
            images, (BATCH_SIZE_DEFAULT, -1))

        # move input data to device if using cluster
        # image_batch_vector, labels = image_batch_vector.to(device), labels.to(device)

        # run model on input data
        outputs = model(image_batch_vector)

        # TODO should have a look because probably doesnt make sense
        # preds = preds.squeeze(dim=1)

        # calculate loss
        # argmax so that loss_module can evaluate output
        labels_argmax = torch.argmax(labels, dim=1)
        # output_argmax = torch.argmax(outputs, dim=1)

        loss = loss_module(outputs, labels_argmax)

        # perform backpropagation
        optimizer.zero_grad()

        # TODO check where to implement loss module in the architecture
        loss.backward()

        # update parameters
        optimizer.step()

        # update eval freq counter
        eval_freq_counter += 1

        # eval frequency check
        if eval_freq_counter == FLAGS.eval_freq or iter_step == FLAGS.max_steps - 1:
            model.eval()
            print(iter_step)
            with torch.no_grad():
                ####### validation #########
                # append training loss
                training_loss.append(loss.item())
            
                # get validation set loss
                # get validation images, labels
                test_images, test_labels = testset.images, testset.labels

                # convert numpy array to torch tensor
                test_images = torch.from_numpy(test_images)
                test_labels = torch.from_numpy(test_labels)

                # need to reshape it into vector so batchsize x (3x32x32)
                test_images_vector = torch.reshape(test_images, (num_examples_test, -1))

                # move input data to device if using cluster
                # val_images_vector, val_labels = val_images_vector.to(device), val_labels.to(device)

                # predictions for validation set
                test_output = model(test_images_vector)

                test_labels_argmax = torch.argmax(test_labels, dim=1)
                # output_argmax = torch.argmax(outputs, dim=1)

                test_loss = loss_module(test_output, test_labels_argmax)

                # append validation loss
                testset_loss.append(test_loss.item())

                ####### accuracy #########
                train_images, train_labels = trainset.images, trainset.labels

                train_images = torch.from_numpy(train_images)
                train_labels = torch.from_numpy(train_labels)

                train_images_vector = torch.reshape(train_images, (num_examples_train, -1))

                train_output = model(train_images_vector)

                train_acc = accuracy(train_output, train_labels)

                training_accuracy.append(train_acc)

                test_acc = accuracy(test_output, test_labels)

                testset_accuracy.append(test_acc)

            # reset counter for next eval frequency
            eval_freq_counter = 0
            
            # put model in training mode again
            model.train()


    # plot the train and validation loss
    fig = plt.figure()
    plt.plot(np.arange(len(training_loss)) * 100, training_loss, label="Training")
    plt.plot(np.arange(len(testset_loss)) * 100, testset_loss, label="Test")
    plt.xlabel("Step Number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()

    # plot accuracy
    fig = plt.figure()
    plt.plot(np.arange(len(training_accuracy)) * 100, training_accuracy, label="Training")
    plt.plot(np.arange(len(testset_accuracy)) * 100, testset_accuracy, label="Test")
    plt.xlabel("Step Number")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()

    # print final test acc
    print(testset_accuracy[-1])

    #######################
    ###### evaluation #####
    #######################

    # # set to eval mode
    # model.eval()
    # true_preds, num_preds = 0., 0.

    # # TODO probably do not need this part
    # with torch.no_grad():
    #     # get validation images, labels
    #     val_images, val_labels = validationset.images, validationset.labels

    #     # convert numpy array to torch tensor
    #     val_images = torch.from_numpy(val_images)
    #     val_labels = torch.from_numpy(val_labels)

    #     # need to reshape it into vector so batchsize x (3x32x32)
    #     val_images_vector = torch.reshape(val_images, (num_examples_valid, -1))

    #     # move input data to device if using cluster
    #     # val_images_vector, val_labels = val_images_vector.to(device), val_labels.to(device)

    #     # predictions for validation set
    #     val_preds = model(val_images_vector)

    #     # get accuracy
    #     # TODO check what the shape of val_labels is: already matrix, or need to make it matrix
    #     acc = accuracy(val_preds, val_labels)

    # print(acc)

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
