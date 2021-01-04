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
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std. 
            The tensor should have the same shape as the mean and std input tensors.
    """
    # sample random vector from normal distribution
    epsilon = torch.randn(size=(std.shape), device=mean.device)
    
    # compute z
    z = mean + (epsilon * std)
    
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    variance = torch.square(torch.exp(log_std))

    KLD = 0.5 * (variance + torch.square(mean) - 1 - torch.log(variance))
    KLD = torch.sum(KLD, dim=1)

    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images.
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    e = torch.tensor(np.e)
    d = torch.tensor(img_shape[1:])
    
    bpd = elbo * torch.log2(e) / torch.prod(d)
    
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/(grid_size+1)
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use scipy's function "norm.ppf" to obtain z values at percentiles.
    # - Use the range [0.5/(grid_size+1), 1.5/(grid_size+1), ..., (grid_size+0.5)/(grid_size+1)] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a sigmoid after the decoder
    # grid_size=5
    # import pdb
    # pdb.set_trace()
    # make percentile range
    percentiles = np.arange(0.5/(grid_size+1), (grid_size+0.5)/(grid_size+1), 1.0/(grid_size+1))

    # obtain z values at percentiles
    z = norm.ppf(percentiles)


    # create mesh_grid
    # grid_x, grid_y = torch.meshgrid(z, z)

    # create list of grid_points to pass to decoder
    coordinates = []
    for y in z:
        for x in z: 
            coordinates.append([x, y])
    
    coordinates = torch.tensor(coordinates, device=decoder.device, dtype=torch.float32)
    output = torch.sigmoid(decoder(coordinates))
    img_grid = make_grid(output, nrow=grid_size)
    # raise NotImplementedError

    return img_grid
