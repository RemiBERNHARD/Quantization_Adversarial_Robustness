#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:32:51 2020

@author: rb258034
"""

# Luring of adversarial perturbations

This repository contains pretrained models and python scripts to reproduce results presented in the article Impact of Low-bitwidth Quantization on the Adversarial Robustness for Embedded Neural Networks](https://arxiv.org/abs/1909.12741).


## Environment and libraries

The python scripts were executed in the following environment:

* OS: CentOS Linux 7
* GPU: NVIDIA GeForce GTX 1080 
* Cuda version: 9.0.176
* Python version: 2.7.5

The following version of some Python packages are necessary: 

* Tensorflow: 1.12.0
* Cleverhans: 3.0.1
* Keras: 2.2.4
* Numpy: 1.16.12


## Files

In order to get the SVHN data set (to have the same training, validation and testing tests, as well as to run attacks), and all pretrained models, download the "models" and "SVHN_data" folders from https://drive.google.com/open?id=167Xo9DNRUCdVtv3bsJdo7kFuLPa1EQrl and place them in the same directory as all other files.


### Model files
    
The "models" repository contains the pretrained models for the datasets SVHN and CIFAR10. As an example, on CIFAR10:    

* models/CIFAR10_float.h5 is the full-precision (32-bit floating point) classifier
* models/CIFAR10_wi_aj.h5 is the classifier with weights quantized to i bits and activation quantized to j bits

The repository contains models with activation quantized to 1 (binary), 2, 3, 4 or 32 (full-precision) bits, and weights quantized to
1 (binary), 2, 3, or 4 bits.

### Evaluation under attack

#### Direct attacks

As an example, to craft adversarial examples on a fully-binarized model (weight and activation quantized to 1 bit) trained on CIFAR10, in order 
to reproduce the results of the part "Robustness against gradient-based and gradient-free attacks":

    python attack_cifar.py w1_a1 0

#### Transferability results

As an example, to craft adversarial examples on a full-precision model and transfer them to a model w2_a2 (weight and activation
quantized to 2 bits) trained on CIFAR10, to reproduce the results of the part "Transferability"::

    python attack_transfer_cifar.py float 0 w2_a2
    
    
    
    