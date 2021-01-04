################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np


class MLPEncoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dims=[512], z_dim=20):
        """
        Encoder with an MLP network and ReLU activations (except the output layer).

        Inputs:
            input_dim - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            z_dim - Dimensionality of latent vector.
        """
        super().__init__()

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.
        self.z_dim = z_dim

        layers = []
        layer_sizes = [input_dim] + hidden_dims
        for idx in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[idx-1], layer_sizes[idx]),
                       nn.ELU()]
        layers += [nn.Linear(layer_sizes[-1], z_dim * 2)]
        self.encoder = nn.Sequential(*layers)

        

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        # Remark: Make sure to understand why we are predicting the log_std and not std
        
        # turn x intor [batch_size, image_vector(784)]
        x_reshape = torch.reshape(x, shape=(x.shape[0], 784))

        encoded_x = self.encoder(x_reshape)

        mean = encoded_x[:, 0:self.z_dim]
        log_std = encoded_x[:, self.z_dim:]
        
        return mean, log_std


class MLPDecoder(nn.Module):

    def __init__(self, z_dim=20, hidden_dims=[512], output_shape=[1, 28, 28]):
        """
        Decoder with an MLP network.
        Inputs:
            z_dim - Dimensionality of latent vector (input to the network).
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of output image. The number of output neurons of the NN must be
                           the product of the shape elements.
        """
        super().__init__()
        self.output_shape = output_shape

        output_shape = np.prod(output_shape)

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.
        layers = []
        layer_sizes = [z_dim] + hidden_dims
        for idx in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[idx-1], layer_sizes[idx]),
                       nn.ELU()]
        layers += [nn.Linear(layer_sizes[-1], output_shape)]
        self.decoder = nn.Sequential(*layers)
        

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,output_shape[0],output_shape[1],output_shape[2]]
        """
        # get batch_size for reshape
        batch_size = z.shape[0]

        # final output shape
        output_shape = tuple([batch_size] + self.output_shape)
        
        # decode z
        decoded_x = self.decoder(z)

        # reshape
        x = torch.reshape(decoded_x, shape=output_shape)
       
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
