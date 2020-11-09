"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()

        self.conv0 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)

        self.preAct1Bn = nn.BatchNorm2d(64)
        self.preAct1ReLU = nn.ReLU()
        self.preAct1Conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=1,)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preAct2aBn = nn.BatchNorm2d(128)
        self.preAct2aReLU = nn.ReLU()
        self.preAct2aConv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)

        self.preAct2bBn = nn.BatchNorm2d(128)
        self.preAct2bReLU = nn.ReLU()
        self.preAct2bConv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preAct3aBn = nn.BatchNorm2d(256)
        self.preAct3aReLU = nn.ReLU()
        self.preAct3aConv = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)

        self.preAct3bBn = nn.BatchNorm2d(256)
        self.preAct3bReLU = nn.ReLU()
        self.preAct3bConv = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=1)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preAct4aBn = nn.BatchNorm2d(512)
        self.preAct4aReLU = nn.ReLU()
        self.preAct4aConv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)

        self.preAct4bBn = nn.BatchNorm2d(512)
        self.preAct4bReLU = nn.ReLU()
        self.preAct4bConv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)

        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preAct5aBn = nn.BatchNorm2d(512)
        self.preAct5aReLU = nn.ReLU()
        self.preAct5aConv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)

        self.preAct5bBn = nn.BatchNorm2d(512)
        self.preAct5bReLU = nn.ReLU()
        self.preAct5bConv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)

        self.pool5 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preFinalBn = nn.BatchNorm2d(512)
        self.preFinalReLU = nn.ReLU()
        self.linear1 = nn.Sequential(nn.Flatten(), nn.Linear(512, n_classes))

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
        x = self.conv0(x)

        save_1 = x
        block1 = self.preAct1Conv(self.preAct1ReLU(self.preAct1Bn(x)))

        out1 = save_1 + block1
        
        out1 = self.conv1(out1)
        out1 = self.pool1(out1)

        save_2 = out1
        block2a = self.preAct2aConv(self.preAct2aReLU(self.preAct2aBn(out1)))
        save_2a = save_2 + block2a
        block2b = self.preAct2bConv(self.preAct2bReLU(self.preAct2bBn(save_2a)))
        
        out2 = save_2a + block2b


        out2 = self.conv2(out2)
        out2 = self.pool2(out2)

        save_3 = out2
        block3a = self.preAct3aConv(self.preAct3aReLU(self.preAct3aBn(out2)))
        save_3a = save_3 + block3a
        block3b = self.preAct3bConv(self.preAct3bReLU(self.preAct3bBn(save_3a)))

        out3 = save_3a + block3b

        out3 = self.conv3(out3)
        out3 = self.pool3(out3)

        save_4 = out3
        block4a = self.preAct4aConv(self.preAct4aReLU(self.preAct4aBn(out3)))
        save_4a = save_4 + block4a
        block4b = self.preAct4bConv(self.preAct4bReLU(self.preAct4bBn(save_4a)))

        out4 = save_4a + block4b

        out4 = self.pool4(out4)

        save_5 = out4
        block5a = self.preAct5aConv(self.preAct5aReLU(self.preAct5aBn(out4)))
        save_5a = save_5 + block5a
        block5b = self.preAct5bConv(self.preAct5bReLU(self.preAct5bBn(save_5a)))

        out5 = save_5a + block5b

        out5 = self.pool5(out5)

        out6 = self.preFinalBn(out5)
        out6 = self.preFinalReLU(out6)

        out = self.linear1(out6)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
