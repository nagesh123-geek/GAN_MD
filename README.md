# Generative Adversarial Networks for Molecular Time series trajectory generation via implicit distribution learning

Welcome to the repository of the research project titles "Efficient Molecular Dynamics Trajectory Sampling using Generative
Adversarial Networks".

# Abstract

Molecular dynamics (MD) simulations provide a microscopic description of molecular motion but
remain severely limited in accessing long timescales and rare events due to their high computational
cost. Majority of generative machine-learning approaches such as diffusion or normalizing flow have
shown promise in accelerating configurational sampling, yet these methods are inherently designed
to generate independent equilibrium snapshots and do not preserve temporal correlations or kinetic
information by construction. On the other hand Autoregressive sequence models, such as LSTM-
and transformer-based architectures, can learn time evolution but generate trajectories step-by-
step, making them susceptible to error accumulation and loss of global dynamical consistency. Here,
we introduce a complementary paradigm inspired by advances in image and video generation: we
treat finite MD trajectory segments as high-dimensional objects and learn their joint distribution
using Generative Adversarial Networks (GANs). Using a Wasserstein GAN with gradient penalty,
we directly generate entire time-series trajectories that are evaluated holistically, enforcing temporal
coherence without explicit integration of equations of motion. We demonstrate the generality of this
approach across systems of increasing complexity, including a two-dimensional three-well model,
protein-ligand binding in cytochrome P450, latent-space dynamics of the intrinsically disordered
protein α-synuclein, and conditional trajectory synthesis for the Trp-cage mini-protein. In all
cases, the generated trajectories reproduce free-energy landscapes and kinetic signatures such as
transition statistics and implied timescales, while enabling rapid generation of rare-event dynamics
that would otherwise require weeks of MD simulation.


# Architecture
![Model architecture](figures/GAN_architecture.png)


# Training and Sampling scheme 

![Training and sampling](figures/GAN_arch.png)


# GAN_MD
This repository contains the GAN models developed for Molecular dynamics data for four systems of study, viz, two dimesnional Brownian Motion in three well potential
Protein Ligand distance trajectory , Trp cage mini protein and alphasynuclein.

# Training and Sampling 
Training and sampling script for various systems are provided in the respective folders

# Data 

In data section , we provided the data for toy two dimensional three well potential Brownian Dynamics for user reproducability.


# Requirements 
pytorch
numpy
matplotlib





