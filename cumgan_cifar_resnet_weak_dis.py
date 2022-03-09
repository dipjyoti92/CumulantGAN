#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:35:53 2018

@author: dipjyoti
"""

import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
import functools
import csv
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = os.path.join(dir_path,'data/cifar-10-batches-py')
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory!')

# -------------------
# input arguments
# -------------------

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--epochs', '-e', type=int, help='[int] how many generator iterations to train for')
parser.add_argument('--disc_iters', '-d', type=int, help='[int] how many discriminator iterations to train per generator')
parser.add_argument('--mode', '-m', type=str, help='[str] wgan-gp')
parser.add_argument('--beta', '-b', type=float, help='[float] cumulantgan beta parameter')
parser.add_argument('--gamma', '-g', type=float, help='[float] cumulantgan gamma parameter')
parser.add_argument('--iteration', '-i', type=int, help='[int] total number of runs')
parser.add_argument('--sess_name', '-s', type=str, help='[str] name of the session run')

args = parser.parse_args()

ITERS = args.epochs
CRITIC_ITERS = args.disc_iters
MODE = args.mode
beta = args.beta
gamma = args.gamma
itr = args.iteration
sess_name = args.sess_name

print('beta value:', beta)
print('gamma value:', gamma)


DIM = 128                    # Model dimensionality
LAMBDA = 10                  # Gradient penalty lambda hyperparameter
BATCH_SIZE = 64              # Batch size
OUTPUT_DIM = 3072            # Number of pixels in CIFAR10 (3*32*32)
G_gp = False                 # Gradient penalty added in Generator
if beta == 0 and gamma == 0:
    gp_1_sided = False       # 2 sided gradient penalty
else:
    gp_1_sided = True        # 1 sided gradient penalty

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def Generator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generatorv2.Input', 128, 2*2*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 2, 2])                    

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')     
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')      
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')     
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')     

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)                                     
    output = tf.reshape(output, [-1, OUTPUT_DIM])
    
    return output


def Discriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)      

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')          
    output = ResidualBlock('Discriminator.Res2', 2*dim, 2*dim, 3, output, resample='down')              

    output = tf.reshape(output, [-1, 4*4*8*dim])                           
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)         
    output = tf.reshape(output, [-1])                                

    return output


real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    if beta == 0:
        disc_cost_real = tf.reduce_mean(disc_real)
    else:
        disc_cost_real = -(1.0 / beta) * tf.log(tf.reduce_mean(tf.exp((-beta) * disc_real)))

    if gamma == 0:
        disc_cost_fake = tf.reduce_mean(disc_fake)
        gen_cost = -tf.reduce_mean(disc_fake)

    else:
        disc_cost_fake = (1.0 / gamma) * tf.log(tf.reduce_mean(tf.exp(gamma * disc_fake)))
        gen_cost = - (1.0 / gamma) * tf.log(tf.reduce_mean(tf.exp(gamma * disc_fake)))

    disc_cost = disc_cost_fake - disc_cost_real

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    # cumulant loss
    if beta == 0:
        disc_cost_real = tf.reduce_mean(disc_real)
    else:
        max_val = tf.reduce_max((-beta) * disc_real)
        disc_cost_real = -(1.0 / beta) * (tf.log(tf.reduce_mean(tf.exp((-beta) * disc_real - max_val))) + max_val)

    if gamma == 0 :
        disc_cost_fake = tf.reduce_mean(disc_fake)
        gen_cost = -tf.reduce_mean(disc_fake)

    else:
        max_val = tf.reduce_max((gamma) * disc_fake)
        disc_cost_fake = (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * disc_fake - max_val))) + max_val)
        gen_cost = - (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * disc_fake - max_val))) + max_val)

    disc_cost = disc_cost_fake - disc_cost_real

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

    if gp_1_sided:
        gradient_penalty = tf.reduce_mean((tf.math.maximum(0.,(slopes-1.)))**2)
        print(""" ========================================== """)
        print('1_sided GP')
        print(""" ========================================== """)
    else:
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        print(""" ========================================== """)
        print('2_sided GP')
        print(""" ========================================== """)

    disc_cost += LAMBDA*gradient_penalty

    if G_gp:
        gen_cost -= LAMBDA*gradient_penalty
        print(""" ========================================== """)
        print('GP in Generator')
        print(""" ========================================== """)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')

# For calculating inception score
samples_100 = Generator(100)
inception_scores, frechet_distances = [],[]

def get_inception_scores(iteration):

    all_samples = []
    for i in range(500):
        all_samples.append(session.run(samples_100))

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return all_samples, lib.inception_score.get_inception_score(list(all_samples))


# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images,_ in train_gen():
            yield images

def inf_dev_gen(dev_gen_frechet):
    while True:
        for images,_ in dev_gen_frechet():
            yield images

real_image_tf = tf.placeholder(dtype=tf.float32,shape=[1000,32,32,3])
all_samples_tf = tf.placeholder(dtype=tf.float32,shape=[1000,32,32,3])
_, dev_gen_frechet = lib.cifar10.load(1000, data_dir=DATA_DIR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

if not os.path.exists('cifar10/'+str(sess_name)+'/'):
    os.makedirs('cifar10/'+str(sess_name)+'/')

# Train loop
with tf.Session(config=config) as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()
    dev = inf_dev_gen(dev_gen_frechet)


    inception_file = 'cifar10/'+str(sess_name)+'/'+MODE+'_inception_scores_beta_' + str(beta) + '_gamma_' + str(gamma)+ '_set'+str(itr) + '.csv'

    if os.path.isfile(inception_file):
        inception_scores = []
        with open(inception_file, "r") as output:
            reader = csv.reader(output, lineterminator='\n')
            for val in reader:
                inception_scores.append(val[0])
    else:
        inception_scores = []

    # Saving and loading checkpoints function

    def save_checkpoint(checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)  
        saver.save(session, os.path.join(checkpoint_dir, 'cumgan'), global_step=step)
        if step % 50000 == 49999:
            best_saver.save(session, os.path.join(checkpoint_dir, 'cumgan_best'), global_step=step)

    def load_checkpoint(checkpoint_dir):
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            return step
        else:
            return 0

    saver = tf.train.Saver(max_to_keep=1)
    best_saver = tf.train.Saver(max_to_keep=6)

    # resuming training from checkpoints

    resume_iter = load_checkpoint('cifar10/'+str(sess_name)+'/checkpoints/')

    if resume_iter > ITERS:
        sys.exit('Iteration must be greater than the resume checkpoint iterations')

    if resume_iter > 0:
        print(""" ========================================== """)
        print("An existing model was found -resuming from checkpoints", resume_iter)
        print(""" ========================================== """)
    else:
    	resume_iter = 0
    	print(""" ========================================== """)
    	print(""" No model found - initializing a new one""")
    	print(""" ========================================== """)

    for iteration in range(resume_iter+1, ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            if G_gp:
                _data = next(gen)
                _ = session.run(gen_train_op, feed_dict={real_data_int: _data})
            else:          
                _ = session.run(gen_train_op)

        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)

        # Calculate inception score
        with tf.device('/device:GPU:0'):
            if iteration % 500 == 499:
                all_samples, inception_score = get_inception_scores(iteration)
                print(""" ========================================== """)
                print("Inception scores: ",inception_score[0])
                print(""" ========================================== """)
                inception_scores.append(inception_score[0])

        # saving inception scores and checkpoints

        if iteration % 500 == 499:
            save_checkpoint('cifar10/'+str(sess_name)+'/checkpoints/', iteration)
            with open(inception_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in inception_scores:
                    writer.writerow([val])

        # Calculate dev loss
        if iteration % 1000 == 999:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images}) 
                dev_disc_costs.append(_dev_disc_cost)
            print('Iteration:', iteration, 'and dev_disc_cost:', np.mean(dev_disc_costs))

        lib.plot.tick()



print("""========================================== """)
print('Finised training the model!!')
print("""========================================== """)
