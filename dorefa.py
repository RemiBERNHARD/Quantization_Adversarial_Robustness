#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:42:07 2019

@author: rb258034
"""

import tensorflow as tf
import functools

#memoized = functools.lru_cache(maxsize=None)

def graph_memoized(func):
    """
    Like memoized, but keep one cache per default graph.
    """

    import tensorflow as tf
    GRAPH_ARG_NAME = '__IMPOSSIBLE_NAME_FOR_YOU__'

    #@memoized
    def func_with_graph_arg(*args, **kwargs):
        kwargs.pop(GRAPH_ARG_NAME)
        return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert GRAPH_ARG_NAME not in kwargs, "No Way!!"
        graph = tf.get_default_graph()
        kwargs[GRAPH_ARG_NAME] = graph
        return func_with_graph_arg(*args, **kwargs)
    return wrapper


@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x, force_quantization=False):
        if bitW == 32 and not force_quantization:
            return x
        if bitW == 1:   # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.tanh(x)
        x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0) # it seems as though most weights are within -1 to 1 region anyways
        return 2*quantize(x, bitW)-1 
        
    def fa(x):
        if bitA == 32:
            return x
        #return quantize_2(x,bitA)
        return quantize(x, bitA)
        

    def fg(x):
        return(x)
        
    return fw, fa, fg
#    @tf.RegisterGradient("FGrad")
#    def grad_fg(op, x):
#        rank = x.get_shape().ndims
#        assert rank is not None
#        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
#        x = x / maxx
#        n = float(2**bitG - 1)
#        x = x * 0.5 + 0.5 + tf.random_uniform(
#            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
#        x = tf.clip_by_value(x, 0.0, 1.0)
#        x = quantize(x, bitG) - 0.5
#        return x * maxx * 2
#
#    def fg(x):
#        if bitG == 32:
#            return x
#        with G.gradient_override_map({"Identity": "FGrad"}):
#            return tf.identity(x)
#    return fw, fa, fg







