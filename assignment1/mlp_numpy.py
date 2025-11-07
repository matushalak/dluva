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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.IN, self.HIDDEN, self.OUT = n_inputs, n_hidden, n_classes
        layer_sizes = [n_inputs] + n_hidden
        self.params = dict()
        # Parameter initialization already handled inside modules
        if n_hidden == 0 or len(n_hidden) == 0 or n_hidden is None or n_hidden[0] == 0:
            # logistic regression edge case
            self.layers = [LinearModule(n_inputs, n_classes, input_layer=True)]
        else:
            # Chain linear layers and ELU nonlinearities
            self.layers = []
            for il, (insize, outsize) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                self.layers.append(LinearModule(insize, outsize, input_layer = il == 0))
                self.layers.append(ELUModule())
        
            # Add final logit head (without ELU nonlinearity)
            self.layers.append(LinearModule(n_hidden[-1], n_classes))
        
        # Here we use the softmax module
        self.softmax = SoftMaxModule()    
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
        # Go through linear layers 
        # and nonlinear activations
        for l in self.layers:
            x = l.forward(x)
        # Apply softmax on logits in output layer
        out = self.softmax.forward(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Gradient of softmax wrt logits given gradient of loss w
        dout = self.softmax.backward(dout)
        for l in self.layers[::-1]:
            dout = l.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for l in self.layers:
            l.clear_cache()
        self.softmax.clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################
