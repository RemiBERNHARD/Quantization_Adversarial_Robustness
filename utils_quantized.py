#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 08:07:47 2020

@author: rb258034
"""

import sys
sys.path.insert(1, '/layers')

from keras.models import load_model
from layers.layers_binary import BinaryDense, BinaryConv2D, BinaryConv2Dweights, Clip
from layers.layers_binaryconnect import BinaryConnectDense, BinaryConnectConv2D
from layers.layers_dorefa_w2_a2 import DorefaDense as DorefaDense_w2_a2, DorefaConv2Dweights as DorefaConv2Dweights_w2_a2, DorefaConv2D as DorefaConv2D_w2_a2
from layers.layers_dorefa_w2_a32 import DorefaDense as DorefaDense_w2_a32, DorefaConv2Dweights as DorefaConv2Dweights_w2_a32, DorefaConv2D as DorefaConv2D_w2_a32
from layers.layers_dorefa_w3_a3 import DorefaDense as DorefaDense_w3_a3, DorefaConv2Dweights as DorefaConv2Dweights_w3_a3, DorefaConv2D as DorefaConv2D_w3_a3
from layers.layers_dorefa_w3_a32 import DorefaDense as DorefaDense_w3_a32, DorefaConv2Dweights as DorefaConv2Dweights_w3_a32, DorefaConv2D as DorefaConv2D_w3_a32
from layers.layers_dorefa_w4_a4 import DorefaDense as DorefaDense_w4_a4, DorefaConv2Dweights as DorefaConv2Dweights_w4_a4, DorefaConv2D as DorefaConv2D_w4_a4
from layers.layers_dorefa_w4_a32 import DorefaDense as DorefaDense_w4_a32, DorefaConv2Dweights as DorefaConv2Dweights_w4_a32, DorefaConv2D as DorefaConv2D_w4_a32

def load_quantized_model(dataset, model_type):
    if (model_type == "float"):
        model = load_model("models/" + dataset + "_" + model_type + ".h5")       
    elif (model_type == "w1_a1"):
        model = load_model("models/" + dataset + "_" + model_type + ".h5",
                           custom_objects={'BinaryConv2Dweights':BinaryConv2Dweights, 'BinaryConv2D':BinaryConv2D,
                                           'BinaryDense':BinaryDense, 'Clip':Clip}) 
    elif (model_type == "w1_a32"):
         model = load_model("models/" + dataset + "_" + model_type + ".h5",
                        custom_objects={'BinaryConnectConv2D':BinaryConnectConv2D, 
                                        'BinaryConnectDense':BinaryConnectDense, 'Clip':Clip})
    else :
        model = load_model("models/" + dataset + "_" + model_type + ".h5",
                           custom_objects={'DorefaConv2D': eval("DorefaConv2D_" + model_type), 
                                    'DorefaConv2Dweights': eval("DorefaConv2Dweights_" + model_type),
                                    'DorefaDense': eval("DorefaDense_" + model_type), 'Clip':Clip})
    return(model)
