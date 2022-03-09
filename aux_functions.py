#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:01:17 2018

@author: dipjyoti
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
metric = "euclidean"


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 0.2 / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def plot(samples, X_mb, scale):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.plot(samples[:,0],samples[:,1],'ro', X_mb[:,0], X_mb[:,1], 'bo')
    plt.axis([-scale, scale, -scale, scale])
    return fig

def plot_mnist(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def plot_withDec(samples, X_mb, dec, scale):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.plot(X_mb[:,0], X_mb[:,1], 'bo')
    for i in range(len(dec)):
#        print dec[i]
        if dec[i]>0.5:
            plt.plot(samples[i,0],samples[i,1],'ro')
        else:
            plt.plot(samples[i,0],samples[i,1],'go')
    plt.axis([-scale, scale, -scale, scale])
    return fig


def mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)

def mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2

def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = mix_rbf_kernel(X, Y, sigmas, wts)
    mmd=mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return mmd


def update_weights_gen(d_z, beta):
#    weights = np.exp(beta*d_z)
    weights = np.exp(beta*np.log(d_z))
    return weights

def update_weights_dis(d_z, beta):
#    weights = np.exp(-beta*d_z)
    weights = np.exp(-beta*np.log(d_z))
    return weights

def update_weights_wass_gen(d_z, beta):    
    weights = np.exp(beta*d_z)
    return weights    

def update_weights_wass_dis(d_z, beta):    
    weights = np.exp(beta*d_z)
    return weights    


def update_weights_iwgan(d_z):
    weights = d_z/(1.001-d_z) # 0.001 was added to avoid divide by zero
    return weights


def mapping_weights(weights, gz1, gz2,mb_size):
    weights_new = np.zeros(shape=(mb_size,1))
    dist = cdist( gz1, gz2, metric=metric )
    imax = np.argmin(dist, axis=1)
    weights_new = weights[imax]
    return weights_new

def reshape_frechet(images, mb_size):
    data = np.zeros(shape=(mb_size,28,28,1))

    for it in range(images.shape[0]):
        x_mb=images[it,:]
        data[it,:,:,0]=x_mb.reshape(28, 28)
        
    return data
