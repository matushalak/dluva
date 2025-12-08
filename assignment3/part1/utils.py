################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


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
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    # reparametrization trick: z = mu + sigma * eps     with eps ~ N(0,I)
    z = mean + (std * torch.randn(mean.shape, device=mean.device))
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    # KLD of 2 multivar gaussians expressed as sum of univar gaussians
    # solely with mean and log_std along each dimension
    KLD:torch.Tensor = 0.5 * (torch.exp(2*log_std) + torch.pow(mean, 2) - 1 - (2*log_std))
    return torch.sum(KLD,dim = -1)


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    # change log base from e to 2
    nll = elbo * torch.log2(torch.tensor(torch.e, device=elbo.device))
    # exclude first batch dim
    image_dims = torch.tensor(img_shape[1:], device = elbo.device)
    # bits per dimension score
    bpd = nll * (1 / torch.prod(image_dims, dim=0))
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
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    # percentile range with significant density
    percentiles = torch.linspace(0.5/grid_size, (grid_size-0.5)/grid_size, grid_size)
    standard_normal = torch.distributions.Normal(loc=0, scale=1)
    zvals = standard_normal.icdf(percentiles)
    z1grid, z2grid = torch.meshgrid(zvals, zvals, indexing='ij')
    # reformat to collapse grid to one batch and match latent dimensionality
    zgrid = torch.stack([z1grid.flatten(), z2grid.flatten()], dim=1) # (grid**2, 2)
    # can pass to decoder (B=grid**2, zdim = 2)
    logits = decoder(zgrid) # (grid**2, 2) => (grid**2, 16, 28, 28)
    probs = torch.softmax(logits, dim = 1)
    # compute output means
    values = torch.arange(16)
    output_means = probs * values[None, :, None, None]
    output_means = output_means.sum(dim = 1).float().unsqueeze(1)
    output_means /= 15
    # make grid with torchvision make_grid
    img_grid = make_grid(output_means, nrow=grid_size, padding=0, normalize=True, value_range=(0,1))

    return img_grid

