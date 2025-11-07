################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs:int, n_hidden:list[int], n_classes:int, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        self.IN, self.HIDDEN, self.OUT = n_inputs, n_hidden, n_classes
        layer_sizes = [n_inputs] + n_hidden
        
        if n_hidden == 0 or len(n_hidden) == 0 or n_hidden is None or n_hidden[0] == 0:
            # logistic regression edge case
            self.layers = nn.Linear(n_inputs, n_classes)
            # equal activation and gradient variance for linear layer without activation function
            nn.init.normal_(self.layers.weight, 0, 1/(self.layers.weight.shape[1]**0.5))
            nn.init.zeros_(self.layers.bias)
        
        else:
            layers = []
            for il, (insize, outsize) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                l = nn.Linear(insize, outsize)
                # Initialization of weights with kaiming, biases with zeros
                if il == 0:
                    # "Linear" Kaiming for first layer without ReLU
                    nn.init.normal_(l.weight, 0, 1/(l.weight.shape[1]**0.5))
                    nn.init.zeros_(l.bias)
                else:
                    # ELU is not supported but Kaming for relu close enough
                    nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
                    nn.init.zeros_(l.bias)
                layers += [l]
                # batch norm before non-linearity
                if use_batch_norm:
                    layers += [nn.BatchNorm1d(outsize)]
                layers += [nn.ELU()]
            
            # Final logit layer - without batch norm and elu
            head = nn.Linear(n_hidden[-1], n_classes)
            nn.init.kaiming_normal_(head.weight, nonlinearity='relu')
            nn.init.zeros_(head.bias)
            layers += [head]
            # Chain linear layers, (batch norms), and ELU nonlinearities
            self.layers = nn.Sequential(*layers)


        #######################
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # nn.Sequential in __init__ describes the forward pass
        out = self.layers(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
