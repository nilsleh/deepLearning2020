"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {}
        self.params['weight'] = np.random.normal(0, 0.0001, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))

        self.grads = {}
        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros((1, out_features))

        self.intermediate = {}
        self.intermediate["weight"] = self.params['weight'].copy()
        self.intermediate["x"] = np.zeros((in_features, 1))

        # raise NotImplementedError

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size = x.shape[0]
        
        output_size = self.params["weight"].shape[0]
     
        out = x @ self.params["weight"].T + np.broadcast_to(self.params["bias"], (batch_size,output_size))

        # store intermediate result
        self.intermediate["x"] = x.copy()
        # self.intermediate["weight"] = self.params["weight"].copy()

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # compute gradient with respect to input of module
        
        dx = dout @ self.params["weight"]
        

        # store gradient updates
        self.grads["weight"] = dout.T @ self.intermediate["x"]
        self.grads['bias'] = np.sum(dout, axis=0)[:, None].T
        # self.grads["bias"] = 


        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
      self.intermediate = np.zeros(1)

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # vectorize max_trick
        
        b = np.max(x, axis=1, keepdims=True)
        x_shift = np.exp(x-b)
        out = x_shift / np.sum(x_shift, axis=1, keepdims=True)

        # b = np.amax(x, 1)
        # temp_x = x - b[:, None]
        # temp_x = np.exp(temp_x)
        # temp_x = np.sum(temp_x, axis=1)
        # temp_x = np.log(temp_x)
        # denominator = b + temp_x
        # out = np.exp(x - denominator[:, None])
       

        self.intermediate = out.copy()
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
       
        inter_step = np.multiply(dout, self.intermediate)
        temp_x = np.einsum("in, in, ij->ij", dout, self.intermediate, self.intermediate)
        
        dx = inter_step - temp_x

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        batch_size = x.shape[0]
        log_x = np.log(x)
        samples_loss = -(1 / batch_size) * np.sum(np.multiply(y, log_x), axis=1)
        out = np.mean(samples_loss)

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
      
        batch_size = x.shape[0]
        reciprocal_x = np.reciprocal(x)

        dx = - (1 / batch_size**2) * np.multiply(y, reciprocal_x)
        
      
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    def __init__(self):
      self.intermediate = np.zeros(1)

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.where(x >= 0, x, np.exp(x) - 1)

        self.intermediate = x.copy()
      
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        elu_derivative = np.where(self.intermediate >= 0, 1, np.exp(self.intermediate))
        dx = np.multiply(dout, elu_derivative)

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
