# Hyperspectral Linear Unmixing with Deep Generative Models

This repository contains a short project made when I was at Mines Paris PSL. It explores the application of small deep learning models to the Hyperspectral Unmixing of Sources on images (HSU) problem. We compare three progressive architectures, focusing on how they handle the essential physical constraints of abundance estimation: Non-Negativity ($\alpha_i \ge 0$) and Sum-to-One ($\sum \alpha_i = 1$). Practical experiments started from the open-source repository based on Palsson et al. (https://github.com/burknipalsson/hu_autoencoders).

## Problem

HSU is an unsupervised representation learning task. It assumes that every pixel in a high-dimensional image is a linear combination of the "purest" source materials (endmembers). The goal is to estimate both the endmembers and their fractional abundances.

- Traditional Baseline: Algorithms like the Simplex Algorithm (e.g., N-FINDR) were historically used to find endmembers.
- AE Approach (2018): Proposed by B. Palsson et al.: Hyperspectral Unmixing Using a Neural Network Autoencoder proposed an Autoencoder (AE) approach. The Autoencoder approach works because the Linear Mixing Model (LMM) can be directly mapped onto the autoencoder architecture: the decoder's weights become the endmember matrix, and the latent code becomes the abundance vector. They achieved strong results using the Angular Reconstruction Loss (Spectral Angle Distance, SAD) that is less sensitive to pixel magnitude than Mean Squared Error.

## Models

- Model 1: Constrained Autoencoder (AE)
    
    This model is a direct neural network implementation of the LMM, with explicit constraints :ReLU for non negativity, and Softplus for normalization, applied to the latent layer.
    
- Model 2: $\beta$-Variational Autoencoder (VAE) with truncated Gaussian Prior
    
    The deterministic AE can be sensitive to image noise, as the mapping from pixel to abundance is direct. To circumvent this, we implement a $\beta$-VAE (original paper Higgins - 2017 - beta-VAE  learning basic visual concepts with a constrained variational framework).
    
    A VAE trades its deterministic latent layer for parameters of a known prior probability distribution ($p(\boldsymbol{\alpha})$). The encoder learns the distribution $q(\boldsymbol{\alpha}|\mathbf{y})$ (typically mean and variance for a Gaussian) to better approximate the true probability distribution, increasing robustness.
    
    The VAE objective adds a KL-Divergence term as a regularization loss:
    $$\mathcal{L}_{\beta-VAE} = \mathcal{L}_{\text{Recon.}} + \mathbf{\beta} \cdot D_{KL}(q(\boldsymbol{\alpha}|\mathbf{y}) \| p(\boldsymbol{\alpha}))$$
        
    The $\beta$ balances reconstruction accuracy against fidelity to the prior in the latent space. The Gaussian prior is simple but limited. Abundances must lie on a simplex (positive and sum to one), whereas the Gaussian has spherical / elliptical density repartition. It seems natural it will struggle to approximate the simplex distribution.
    .
- Model 3: $\beta$-VAE with Dirichlet Latent Prior (Joo and al. - 2020 - Dirichlet variational autoencoder)
    
    This is the most advanced model, using a more suited prior. The Dirichlet distribution is mathematically suitable because it is the natural probability distribution for data that is positive and sums to one. Using it as the prior $p(\boldsymbol{\alpha})$ harmonizes the statistical model with the physical constraints of HSU. The implementation, relying on PyTorch's recent updates for sampling, utilizes the property that a Dirichlet random variable can be approximated and reparameterized from Gamma-distributed random variables.