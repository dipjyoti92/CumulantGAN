#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:35:53 2018

@author: dipjyoti
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from aux_functions import *

import csv
import sys
import argparse

# -------------------
# input arguments
# -------------------

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--epochs', '-e', type=int, help='[int] how many generator iterations to train for')
parser.add_argument('--disc_iters', '-d', type=int, help='[int] how many discriminator iterations to train per generator')
parser.add_argument('--beta', '-b', type=float, help='[float] cumulantgan beta parameter')
parser.add_argument('--gamma', '-g', type=float, help='[float] cumulantgan gamma parameter')
parser.add_argument('--iteration', '-i', type=int, help='[int] total number of runs')
parser.add_argument('--sess_name', '-s', type=str, help='[str] name of the session run')

args = parser.parse_args()

epochs = args.epochs
disc_iters = args.disc_iters
beta = args.beta
gamma = args.gamma
iteration = args.iteration
sess_name = args.sess_name
print('beta value:', beta)
print('gamma value:', gamma)

mb_size = 1000                # Batch size
D_h1_dim=32                   # discriminator hidden dimensions
D_h2_dim=32                   # discriminator hidden dimensions
D_h3_dim=32                   # discriminator hidden dimensions
Z_dim = 8                     # Gaussian noise dimension
G_h1_dim=32                   # generator hidden dimensions
G_h2_dim=32                   # generator hidden dimensions
G_h3_dim=32                   # generator hidden dimensions
X_dim=2                       # data dimension


# ---------------------------------------
# Discriminator parameters initialization
# ---------------------------------------

X = tf.placeholder(tf.float32, shape=[None, 2])

D_W1 = tf.Variable(xavier_init([X_dim, D_h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[D_h1_dim]))

D_W2 = tf.Variable(xavier_init([D_h1_dim, D_h2_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[D_h2_dim]))

D_W3 = tf.Variable(xavier_init([D_h2_dim, D_h3_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[D_h3_dim]))

D_W4 = tf.Variable(xavier_init([D_h3_dim, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

# ---------------------------------------
# Generator parameters initialization
# ---------------------------------------

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, G_h1_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[G_h1_dim]))
    
G_W2 = tf.Variable(xavier_init([G_h1_dim, G_h2_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[G_h2_dim]))

G_W3 = tf.Variable(xavier_init([G_h2_dim, G_h3_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[G_h3_dim]))

G_W4 = tf.Variable(xavier_init([G_h3_dim, X_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

# ----------
# Generator 
# ----------

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_log_prob = tf.matmul(G_h3, G_W4) + G_b4
    return G_log_prob

# ---------------
# Discriminator 
#----------------

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    out = tf.matmul(D_h3, D_W4) + D_b4
    return out

def sample_Z(m, n):
    return np.random.normal(0., 1, size=[m, n])

G_sample = generator(Z)

D_real = discriminator(X)
D_fake = discriminator(G_sample)

# -------------------
# Cumualant losses:
# -------------------

if beta == 0:
    D_loss_real = tf.reduce_mean(D_real)
else:
    max_val = tf.reduce_max((-beta) * D_real)
    D_loss_real = -(1.0 / beta) * (tf.log(tf.reduce_mean(tf.exp((-beta) * D_real - max_val))) + max_val)

if gamma == 0:
    D_loss_fake = tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

else:
    max_val = tf.reduce_max((gamma) * D_fake)
    D_loss_fake = (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)
    max_val = tf.reduce_max((gamma) * D_fake)
    G_loss = - (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

D_loss = D_loss_real - D_loss_fake

# -------------------
# optimizers
# -------------------

D_solver = tf.train.AdamOptimizer().minimize(-D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

if (beta == 0) & (gamma == 0):
    clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]
else:
    clip_D = theta_D

# -------------------
# load data
# -------------------

toy_data = genfromtxt('./data/swiss_roll_2d_with_labels.csv', delimiter=',', dtype='float32')
toy_data = toy_data[:,:-1]


NoT = 500
idx = np.random.randint(toy_data.shape[0], size=toy_data.shape[0])
i = int(toy_data.shape[0]) - NoT

toy_data_train = toy_data[idx[:i], :]
toy_data_test = toy_data[idx[i:], :]

config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# create log folders

csv_name = sess_name + '/csv_D_' + str(D_h1_dim) + '_' + str(D_h2_dim) + '_' + str(D_h3_dim) + '_G_' + str(G_h1_dim) + '_' + str(G_h2_dim) + '_' + str(G_h3_dim) + '/'

if not os.path.exists(csv_name):
    os.makedirs(csv_name)
if not os.path.exists(csv_name+'/plots'):
    os.makedirs(csv_name+'/plots')

# -------------------
# training
# -------------------

i = 0
SF = 1000
D_loss_plots = np.ones(shape=(epochs, 1))
G_loss_plots = np.ones(shape=(epochs, 1))
mmd_plot = np.ones(shape=((int(epochs/SF)),1))

for it in range(epochs):

    # write logs and save samples    

    if it % SF == 0:
        X_mb=toy_data_test
        Z_new = sample_Z(NoT, Z_dim)
        samples = sess.run(G_sample, feed_dict={Z: Z_new})
        dis_dec = sess.run(D_real, feed_dict={X: samples})
        fig = plot_withDec(samples,X_mb,dis_dec,20)
        plt.savefig(csv_name+'/plots/plots{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)

        with open(csv_name+'cumgan_samples_beta_'+str(beta)+'_gamma_'+str(gamma)+'_iteration_'+str(it)+'.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in samples:
                writer.writerow(val)
        i += 1

    # -------------------
    # Update D network
    # -------------------

    for j in range(disc_iters):
        X_mb = toy_data_train[np.random.randint(toy_data_train.shape[0], size=mb_size), :]

        _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict={
                                         X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    # -------------------
    # Update G network
    # -------------------

    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                  X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    # -------------------
    # write loss values
    # -------------------

    if it % SF == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

    D_loss_plots[it] = D_loss_curr
    G_loss_plots[it] = G_loss_curr
    

print("""========================================== """)
print('Finised training the model!!')
print("""========================================== """)