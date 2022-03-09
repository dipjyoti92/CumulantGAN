Cumulant GAN
=====================================

Code for reproducing experiments in ["Cumulant GAN"](https://arxiv.org/abs/2006.06625).

Abstract- In this paper, we propose a novel loss function for training Generative Adversarial Networks (GANs) aiming towards deeper theoretical understanding as well as improved stability and performance for the underlying optimization problem. The new loss function is based on cumulant generating functions giving rise to Cumulant GAN. Relying on a recently-derived variational formula, we show that the corresponding optimization problem is equivalent to Rényi divergence minimization, thus offering a (partially) unified perspective of GAN losses: the Rényi family encompasses Kullback-Leibler divergence (KLD), reverse KLD, Hellinger distance and χ2-divergence. Wasserstein GAN is also a member of cumulant GAN. In terms of stability, we rigorously prove the linear convergence of cumulant GAN to the Nash equilibrium for a linear discriminator, Gaussian distributions and the standard gradient descent ascent algorithm. Finally, we experimentally demonstrate that image generation is more robust relative to Wasserstein GAN and it is substantially improved in terms of both inception score and Fréchet inception distance when both weaker and stronger discriminators are considered.


## Prerequisites

- Python, NumPy, TensorFlow 2, SciPy, Matplotlib

## Models

- `python cumgan_gmm8.py --epochs 10000 --disc_iters 5 --beta 0 --gamma 0 --iteration 0 --sess_name gmm8`: Toy dataset (8 Gaussians). 
- `python cumgan_swissroll.py --epochs 10000 --disc_iters 5 --beta 0 --gamma 0 --iteration 0 --sess_name gmm8`: Toy dataset (Swiss roll). 
- `python cumgan_cifar_weak_dis.py --epochs 200000 --disc_iters 5 --mode wgan-gp --beta 0 --gamma 0 --iteration 0 --sess_name cifar_weak_conv_000`: CIFAR10 data
- `python cumgan_imagenet_resnet_weak_dis.py --epochs 500000 --disc_iters 5 --mode wgan-gp --beta 0 --gamma 0 --iteration 0 --sess_name imagenet_weak_resnet_000`: ImageNET data

The notebook `simulation.ipynb` contains some experimantal designs.

## DATASETS

All the toy data exmaples are in `data` directory. For publicly avaiable CIFAR10 and ImageNET data, you may download and store it inside `data` directory. Otherwise, you can change the data paths inside .py files.
