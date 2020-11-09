"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

# TODO CHANGE BACK UTILS FILES WITH PATHS


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
      
        super().__init__()
                           
        layers = []
        layer_sizes = [n_inputs] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[idx-1], layer_sizes[idx]),
                       # nn.BatchNorm1d(layer_sizes[idx]),
                       nn.ELU()]
        layers += [nn.Linear(layer_sizes[-1], n_classes)]
        self.netLayers = nn.Sequential(*layers)

        # self.netLayers = nn.ModuleList()
        # layer_sizes = [n_inputs] + n_hidden
        # for idx in range(1, len(layer_sizes)):
        #     self.netLayers += [nn.Linear(layer_sizes[idx-1], layer_sizes[idx]),
        #                nn.ELU()]
        # self.netLayers += [nn.Linear(layer_sizes[-1], n_classes)]
    

        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # for layer in self.netLayers:
        #   #print(x.shape)
        #   x = layer(x)

        # out = x
        out = self.netLayers(x)
  
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
