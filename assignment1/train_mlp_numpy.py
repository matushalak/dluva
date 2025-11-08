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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Predicted class = argmax of softmax probabilities
    acc = (np.argmax(predictions, axis = 1) == targets).mean()
    #######################
    # END OF YOUR CODE    #
    #######################

    return acc


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    nsamples = 0
    cum_acc = 0
    # accumulate running accuracy over batches (independent of batch size)
    for Xb, Yb in data_loader:
        Xb = Xb.reshape(-1, model.IN)
        acc = accuracy(model.forward(Xb), Yb)
        batchsize = Yb.shape[0]
        cum_acc += acc * batchsize
        nsamples += batchsize
    avg_accuracy = cum_acc / nsamples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)


    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # get first batch
    X1, Y1 = next(iter(cifar10_loader['train']))
    bsize, rgb, height, width = X1.shape
    nclasses = len(np.unique(Y1))
    insize = rgb * height * width

    # Initialize model and loss module
    model = MLP(n_inputs=insize, n_hidden=hidden_dims, n_classes=nclasses)
    loss_module = CrossEntropyModule()

    best_val = 0
    best_model = None
    
    loss_curve = []
    train_accuracies = []
    val_accuracies = []
    
    def zero_grad(model:MLP):
        for l in model.layers:
            if hasattr(l, 'grads'):
                for g in l.grads.values():
                    # fill all gradients with zeros
                    g[...] = 0
    
    def grad_step(model:MLP, lr:float):
        for l in model.layers:
            if hasattr(l, 'params'):
                for k, p in l.params.items():
                    # Gradient descent on params
                    # (using gradients calculated in backprop)
                    p -= lr * l.grads[k]

    # Training loop including validation and manual SGD
    for epoch in range(epochs):
        for Xtr, Ytr in cifar10_loader['train']:
            Xtr = Xtr.reshape(-1, model.IN)
            # reset gradients
            zero_grad(model)
            # model predictions
            model.clear_cache()
            yhat = model.forward(Xtr)
            # CE loss calculated on softmax
            L = loss_module.forward(yhat, Ytr)
            loss_curve.append(L)
            # backprop
            # gradient of loss wrt softmax probabilities
            dout = loss_module.backward(yhat, Ytr)
            model.backward(dout)
            # Minibatch SGD step
            grad_step(model, lr)
        
        # At the end of epoch, get training and validation accuracy
        ta = evaluate_model(model, cifar10_loader['train'])
        va = evaluate_model(model, cifar10_loader['validation'])

        print(f'Epoch {epoch} accuracy, training: {ta}, validation: {va}')
        train_accuracies.append(ta); val_accuracies.append(va)

        # Save best model
        if va >= best_val:
            best_val = va
            best_model:MLP = deepcopy(model)
    
    # Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'Test accuracy of best model: {test_accuracy}')

    # Add any information you might want to save for plotting
    logging_dict = {
        'Epochs':np.arange(epochs),
        'Train loss': loss_curve,
        'Train acc':train_accuracies,
        'Val acc': val_accuracies,
        'Test acc': test_accuracy
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')
    parser.add_argument('--plot', action = 'store_true')

    args = parser.parse_args()
    kwargs = vars(args)

    plt_flag = kwargs['plot']
    del kwargs['plot']

    if plt_flag:
        import matplotlib.pyplot as plt
        model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
        f1, ax1 = plt.subplots(ncols=2)
        # Loss curve
        ax1[0].plot(logging_dict['Train loss'], label = 'Training loss')
        ax1[0].legend(loc = 1)
        ax1[0].set_ylabel('CE Loss'); ax1[0].set_xlabel(f'Iteration ({kwargs['epochs']} epochs)')
        # Accuracy curve
        ax1[1].plot(logging_dict['Train acc'], label = 'Training set')
        ax1[1].plot(logging_dict['Val acc'], label = 'Validation set')
        ax1[1].legend(loc = 2)
        ax1[1].set_ylabel('Classification Accuracy'); ax1[1].set_xlabel(f'Epoch')
        f1.tight_layout(); plt.show()

    else:
        train(**kwargs)
    