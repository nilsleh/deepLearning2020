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
    batch_size, n_classes = predictions.shape

    prediction_argmax = np.argmax(predictions, axis=1)

    prediction_matrix = np.eye(np.max(prediction_argmax)+1)[prediction_argmax]

    result = prediction_matrix * targets

    correct = np.sum(np.sum(result, axis=1))

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

    # num_inputs = 10
    # batch_size = 5
    # num_classes = 3

    # model = MLP(num_inputs, [15, 20], num_classes)

    # loss_module = CrossEntropyModule()    
    # train_x = np.random.randint(5,10, (batch_size,num_inputs))
    # train_labels = np.random.randint(num_classes, size=batch_size).reshape(-1)
    # one_hot_label = np.eye(num_classes)[train_labels]


    data_dict = cifar10_utils.get_cifar10(
        data_dir=FLAGS.data_dir, validation_size=0)
    trainset = data_dict["train"]
    validationset = data_dict["validation"]
    testset = data_dict["test"]

    # num examples train
    num_examples_train = trainset.num_examples
    print(num_examples_train)

    # num examples validation
    num_examples_valid = validationset.num_examples

    # num examples test
    num_examples_test = testset.num_examples
    print(num_examples_test)

    n_input = 3 * 32 * 32
    n_hidden = [100]
    n_classes = 10

    eta = FLAGS.learning_rate
    # eta = 1e-1

    model = MLP(n_input, dnn_hidden_units, n_classes)
    loss_module = CrossEntropyModule() 

    batch_size = FLAGS.batch_size
    
    trainset_loss = []
    testset_loss = []

    training_accuracy = []
    testset_accuracy = []
    
    iter_num = 5
    eval_freq_counter = 0

    for i in range(FLAGS.max_steps):
        train_images, train_labels = trainset.next_batch(batch_size)
        train_vector = np.reshape(train_images, (batch_size, -1))
        output = model.forward(train_vector)

        # compute loss
        train_loss = loss_module.forward(output, train_labels)
        
        # start backpropagation

        backprop_loss = loss_module.backward(output, train_labels)

        model.backward(backprop_loss)

        # update parameters
        for idx in range(0, len(model.layers), 2):
            linear_layer = model.layers[idx]
            linear_layer.params["weight"] -= eta * linear_layer.grads["weight"]
            linear_layer.params["bias"] -= eta * linear_layer.grads["bias"]

        # update eval freq counter
        eval_freq_counter += 1

        if eval_freq_counter == FLAGS.eval_freq:
            # keep track loss
            test_images, test_labels = testset.images, testset.labels
            test_vector = np.reshape(test_images, (num_examples_test, -1))
            test_output = model.forward(test_vector) 

            output_loss = loss_module.forward(test_output, test_labels)
            testset_loss.append(output_loss)
            trainset_loss.append(train_loss)

            # keep track accuracy
            train_images, train_labels = trainset.images, trainset.labels
            train_vector = np.reshape(train_images, (num_examples_train, -1))

            train_output = model.forward(train_vector)

            train_acc = accuracy(train_output, train_labels)

            test_acc = accuracy(test_output, test_labels)

            training_accuracy.append(train_acc)

            testset_accuracy.append(test_acc)

            # reset counter for next eval frequency
            eval_freq_counter = 0

    # plot the train and validation loss
    fig = plt.figure()
    plt.plot(np.arange(len(trainset_loss)) * 100, trainset_loss, label="Training")
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