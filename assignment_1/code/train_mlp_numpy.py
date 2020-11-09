"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
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


def accuracy(predictions, targets):
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
    batch_size, _ = predictions.shape

    prediction_argmax = np.argmax(predictions, axis=1)
    targets_argmax = np.argmax(targets, axis=1)

    correct = np.sum(prediction_argmax == targets_argmax)

    accuracy = correct / float(batch_size)

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

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    np.random.seed(0)


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

    eta = FLAGS.learning_rate

    model = MLP(n_input, dnn_hidden_units, n_classes)

    loss_module = CrossEntropyModule() 

    train_batch_size = FLAGS.batch_size
    
    training_loss = []
    testset_loss = []

    training_accuracy = []
    testset_accuracy = []

    running_loss = 0.0
    running_acc = 0.0

    eval_frequency = FLAGS.eval_freq

    for iter_step in range(FLAGS.max_steps):
        train_images, train_labels = trainset.next_batch(train_batch_size)
        train_vector = np.reshape(train_images, (train_batch_size, -1))

        train_output = model.forward(train_vector)

        # compute loss
        train_loss = loss_module.forward(train_output, train_labels)
        
        # start backpropagation
        backprop_loss_module = loss_module.backward(train_output, train_labels)

        model.backward(backprop_loss_module)

        # update parameters
        for idx in range(0, len(model.layers), 2):
            linear_layer = model.layers[idx]
            linear_layer.params["weight"] -= eta * linear_layer.grads["weight"]
            linear_layer.params["bias"] -= eta * linear_layer.grads["bias"]

        # update running loss
        running_loss += train_loss

        # running accuracy
        running_acc += accuracy(train_output, train_labels)

        if iter_step % eval_frequency == eval_frequency - 1:
            current_loss = running_loss / (eval_frequency)
            print('[%d] loss: %.3f' %
                  (iter_step + 1, current_loss))
            training_loss.append(current_loss)
            running_loss = 0.0

            # training acc
            current_acc = running_acc / (eval_frequency)
            print('[%d] accuracy: %.3f' %
                  (iter_step + 1, current_acc))
            training_accuracy.append(current_acc)
            running_acc = 0.0

            # keep track testset loss and accuracy
            num_examples_test = testset.num_examples
            test_images, test_labels = testset.images, testset.labels
            test_vector = np.reshape(test_images, (num_examples_test, -1))
            test_output = model.forward(test_vector) 
            test_loss = loss_module.forward(test_output, test_labels)
            test_acc = accuracy(test_output, test_labels)

            testset_loss.append(test_loss)
            testset_accuracy.append(test_acc)


    # plot the train and validation loss
    fig = plt.figure()
    plt.plot(np.arange(len(training_loss)) * eval_frequency + eval_frequency, training_loss, label="Training")
    plt.plot(np.arange(len(testset_loss)) * eval_frequency + eval_frequency, testset_loss, label="Test")
    plt.xlabel("Step Number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()
    fig.savefig("./results/numpyMLP_loss.png")

    # plot accuracy
    fig = plt.figure()
    plt.plot(np.arange(len(training_accuracy)) * eval_frequency + eval_frequency, training_accuracy, label="Training")
    plt.plot(np.arange(len(testset_accuracy)) * eval_frequency + eval_frequency, testset_accuracy, label="Test")
    plt.xlabel("Step Number")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.show()
    fig.savefig("./results/numpyMLP_accuracy.png")

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