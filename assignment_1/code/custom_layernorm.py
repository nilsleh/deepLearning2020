import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of layer normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomLayerNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the layer norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """
    
    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomLayerNormAutograd object.
        
        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        
        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomLayerNormAutograd, self).__init__()
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.n_neurons = n_neurons
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))

        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, input):
        """
        Compute the layer normalization
        
        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: layer-normalized tensor
        
        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement layer normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size, num_neurons = input.shape

        assert self.n_neurons == num_neurons, "{} channels are needed but there are only {}".format(self.n_neurons, num_neurons)

        # compute mean
        mu = torch.mean(input, dim=1, keepdim=True)

        # compute var
        var = torch.var(input, dim=1, unbiased=False, keepdim=True)
        var = (var+self.eps)**(-1./2.)

        # normalize
        X_norm = input - mu
        X_norm = X_norm * var

        # scale and shift
        gamma_matrix = self.gamma.expand(batch_size, -1)
        beta_matrix = self.beta.expand(batch_size, -1)

        out = gamma_matrix * X_norm + beta_matrix
        

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomLayerNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the layer norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomLayerNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomLayerNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """
    
    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the layer normalization
        
        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: layer-normalized tensor
    
        TODO:
          Implement the forward pass of layer normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the backward pass or to be stored
          for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        batch_size, num_neurons = input.shape

        # compute mean
        mu = torch.mean(input, dim=1, keepdim=True)

        # compute variance
        var = torch.var(input, dim=1, unbiased=False, keepdim=True)
        var = (var+eps)**(-1./2.)

        # normalize
        X_norm = input - mu
        X_norm = X_norm * var

        # scale and shift
        gamma_matrix = gamma.expand(batch_size, -1)
        beta_matrix = beta.expand(batch_size, -1)

        out = gamma_matrix * X_norm + beta_matrix
        
        ctx.save_for_backward(input, gamma, beta)

        ctx.variance = var
        ctx.mu = mu
        ctx.X_norm = X_norm

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the layer normalization.
        
        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments
        
        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        X, gamma, beta = ctx.saved_tensors
        grad_input = grad_gamma = grad_beta = None

        batch_size, num_neurons = grad_output.shape
        
        if ctx.needs_input_grad[0]:
          
          first = grad_output * torch.mm(ctx.variance, gamma[:, None].T)

          second = torch.sum(first, axis=1, keepdim=True).expand(-1, num_neurons) / num_neurons

          var_cubed = ctx.variance**3
          x_minus_mu = X - ctx.mu.expand(-1, num_neurons)

          third_prod = grad_output * torch.mm(var_cubed, gamma[:, None].T) * x_minus_mu
          third_sum = torch.sum(third_prod, axis=1, keepdim=True).expand(-1, num_neurons) / num_neurons
          third = (x_minus_mu * third_sum)
          
          grad_input = first - second - third

        if ctx.needs_input_grad[1]:
          grad_gamma = torch.sum(grad_output * ctx.X_norm, dim=0)
  
        if ctx.needs_input_grad[2]:
          grad_beta = torch.sum(grad_output, dim=0)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomLayerNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the layer norm operation for MLPs.
    In self.forward the functional version CustomLayerNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """
    
    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomLayerNormManualModule object.
        
        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        
        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomLayerNormManualModule, self).__init__()
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.n_neurons = n_neurons
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, input):
        """
        Compute the layer normalization via CustomLayerNormManualFunction
        
        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: layer-normalized tensor
        
        TODO:
          Check for the correctness of the shape of the input tensor.
          Instantiate a CustomLayerNormManualFunction.
          Call it via its .apply() method.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size, num_neurons = input.shape

        assert self.n_neurons == num_neurons, "{} channels are needed but there are only {}".format(self.n_neurons, num_neurons)
        
        custom_layer_norm = CustomLayerNormManualFunction()
        out = custom_layer_norm.apply(input, self.gamma, self.beta, self.eps)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out


if __name__ == '__main__':
    # create test batch
    n_batch = 8
    n_neurons = 128
    # create random tensor with variance 2 and mean 3
    x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10
    print('Input data:\n\tmeans={}\n\tvars={}'.format(x.mean(dim=1).data, x.var(dim=1).data))
    
    # test CustomLayerNormAutograd
    print('3.1) Test automatic differentation version')
    bn_auto = CustomLayerNormAutograd(n_neurons)
    y_auto = bn_auto(x)
    print('\tmeans={}\n\tvars={}'.format(y_auto.mean(dim=1).data, y_auto.var(dim=1).data))
    
    # test CustomLayerNormManualFunction
    # this is recommended to be done in double precision
    print('3.2 b) Test functional version')
    input = x.double()
    gamma = torch.sqrt(10 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True))
    beta = 100 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
    bn_manual_fct = CustomLayerNormManualFunction(n_neurons)
    y_manual_fct = bn_manual_fct.apply(input, gamma, beta)
    print('\tmeans={}\n\tvars={}'.format(y_manual_fct.mean(dim=1).data, y_manual_fct.var(dim=1).data))
    # gradient check
    grad_correct = torch.autograd.gradcheck(bn_manual_fct.apply, (input, gamma, beta))
    if grad_correct:
        print('\tgradient check successful')
    else:
        raise ValueError('gradient check failed')
    
    # test CustomLayerNormManualModule
    print('3.2 c) Test module of functional version')
    bn_manual_mod = CustomLayerNormManualModule(n_neurons)
    y_manual_mod = bn_manual_mod(x)
    print('\tmeans={}\n\tvars={}'.format(y_manual_mod.mean(dim=1).data, y_manual_mod.var(dim=1).data))
