"""
This file contains classes for a bimodal Gaussian distribution and a
multivariate Gaussian distribution with diagonal covariance matrix.

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import numpy as np
import torch
import math


def broadcast(x, a, b):
    """
    Broadcast shape of input tensors a and b to be able to perform element-wise
    multiplication along the last dimension of x.
    Inputs:
    x - Input tensor of shape [n, n, d].
    a - First input tensor of shape [d].
    b - Second input tensor of shape [d].

    Returns:
        Tensor of shape [1, 1, d]
    """
    #####
    new_a = a.view(((1,) * (len(x.shape)-1)) + x.shape[-1:])
    new_b = b.view(((1,) * (len(x.shape)-1)) + x.shape[-1:])
    #####
    return new_a, new_b
    # return (t.view(((1,) * (len(x.shape)-1)) + x.shape[-1:]) for t in [a, b])


class BimodalGaussianDiag:
    """
    Class specifying a Bimodal Bivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and to
    sample from the distribution.

    Inputs:
        mu (list)    - List of tensors of shape of 1xdims. These are
                       the mean values of the distribution for each
                       random variable.
        sigma (list) - List of tensors of shape 1xdims. These are the
                       values of standard devations of each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        # TODO: Implement initalization
        self.p1 = MultivariateGaussianDiag(mu[0], sigma[0], dims)
        self.p2 = MultivariateGaussianDiag(mu[1], sigma[1], dims)
    
        # self.mus = torch.tensor(mu)
        # self.sigmas = torch.tensor(sigma)
        self.dims = dims
        # raise NotImplementedError

    def log_prob(self, x):
        # TODO: Implement log probability computation
        logp_p1 = self.p1.log_prob(x)
        logp_p2 = self.p2.log_prob(x)
        logp = torch.log(0.5 * logp_p1.exp() + 0.5 * logp_p2.exp())
        
        # raise NotImplementedError
        return logp

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        
        mask = torch.randint(0,2, size=[num_samples], dtype=torch.bool)

        p1_samples = self.p1.sample(num_samples)
        p2_samples = self.p2.sample(num_samples)
        
        samples = torch.cat((p1_samples[mask, :], p2_samples[~mask, :]), dim=0)

        # raise NotImplementedError
        return samples


class MultivariateGaussianDiag:
    """
    Class specifying a Multivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and
    sample from the distribution.

    Inputs:
        mu (list)    - List of tensors of shape of 1xdims. These are
                       the mean values of the distribution for each
                       random variable.
        sigma (list) - List of tensors of shape 1xdims. These are the
                       values of standard devations of each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        super().__init__()
        # TODO: Implement initalization
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.dims = dims
        # logp = None
        # raise NotImplementedError
        

    def log_prob(self, x):
        # TODO: Implement log probability computation
        # print("Multivariate log prob x shape")
        # print(x.shape)
        broad_mu, broad_sigma = broadcast(x, self.mu, self.sigma)
        exponent = (x-broad_mu) * torch.reciprocal(broad_sigma) * (x-broad_mu)
        log_p = -0.5*self.dims - 0.5 * torch.log(torch.prod(self.sigma)) - 0.5*exponent
        
        log_p = torch.sum(log_p, dim=-1)

        return log_p

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        
        epsilon = torch.randn(num_samples, self.dims)
        samples = self.mu[None, :] + epsilon * self.sigma[None, :]
        # raise NotImplementedError
        return samples
