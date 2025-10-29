import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Gamma

from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import io as sio
import os
import numpy as np
from numpy.linalg import inv
import warnings
import matplotlib
import matplotlib.pyplot as plt


device = torch.device("mps")

############################################################
# Deep Unmixing Autoencoder (DAEU) - PyTorch Implementation
# Implementation of
# B. Palsson, J. Sigurdsson, J. R. Sveinsson and M. O. Ulfarsson, "Hyperspectral Unmixing Using a Neural Network Autoencoder," in IEEE Access, vol. 6, pp. 25646-25656, 2018, doi: 10.1109/ACCESS.2018.2818280.


class SumToOneSimple(nn.Module):
    def __init__(self):
        super(SumToOneSimple, self).__init__()

    def forward(self, x):
        # Normalize to sum to 1 (ASC constraint)
        x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-7)
        return x


class SparseReLU(nn.Module):
    def __init__(self, num_features):
        super(SparseReLU, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return F.relu(x - self.alpha)


class Autoencoder(nn.Module):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.params = params
        self.is_deep = True
        self.device = device

        n_end = params["num_endmembers"]
        n_bands = params["n_bands"]

        # Encoder layers
        if self.is_deep:
            self.encoder = nn.Sequential(
                nn.Linear(n_bands, n_end * 9, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 9, n_end * 6, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 6, n_end * 3, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 3, n_end, bias=False),
                nn.LeakyReLU(0.1),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_bands, n_end, bias=False), nn.LeakyReLU(0.1)
            )

        # Utility layers
        self.batch_norm = nn.BatchNorm1d(n_end)
        self.sparse_relu = SparseReLU(n_end)
        self.sum_to_one = SumToOneSimple()
        self.dropout = nn.Dropout(p=0.0045)

        # Decoder layer (endmembers)
        self.decoder = nn.Linear(n_end, n_bands, bias=False)

        # Initialize decoder weights to be non-negative
        with torch.no_grad():
            self.decoder.weight.data = torch.abs(self.decoder.weight.data)

        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)  # Encoder: project input to latent space
        encoded = self.batch_norm(encoded)  # Batch normalization: stabilize training
        encoded = self.sparse_relu(encoded)  # Soft thresholding: enforce sparsity
        abundances = self.sum_to_one(
            encoded
        )  # Sum to one constraint: ensure abundances sum to 1
        abundances_dropped = self.dropout(abundances)  # Dropout: regularize abundances
        decoded = self.decoder(
            abundances_dropped
        )  # Decoder: reconstruct input from abundances
        return decoded, abundances

    def fit(self, data, plot_every=0):
        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)

        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        optimizer = self.params["optimizer"]
        loss_fn = self.params["loss"]

        history = []
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                outputs, abundances = self.forward(batch_x)

                loss = loss_fn(outputs, batch_y)

                loss.backward()

                # ensure decoder weights are non-negative
                with torch.no_grad():
                    self.decoder.weight.data = torch.clamp(
                        self.decoder.weight.data, min=0
                    )

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)

            if plot_every > 0 and (epoch + 1) % plot_every == 0:
                print(
                    f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {avg_loss:.6f}"
                )

        return history

    def get_endmembers(self):
        # transpose to get shape (num_endmembers, n_bands) instead of (n_bands, num_endmembers)
        return self.decoder.weight.data.cpu().numpy().T

    def get_abundances(self):
        self.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(self.params["data"].array()).to(self.device)
            _, abundances = self.forward(data_tensor)
            abundances_np = abundances.cpu().numpy()

            abundances_reshaped = np.reshape(
                abundances_np,
                [
                    self.params["data"].cols,
                    self.params["data"].rows,
                    self.params["num_endmembers"],
                ],
            )

        self.train()
        return abundances_reshaped


class BetaVAE(Autoencoder):
    """
    Beta-Variational Autoencoder with multiple latent distribution options.

    Args:
        params (dict): A dictionary of parameters. Must include:
            - 'latent_dist' (str): The latent distribution to use.
              Options: 'gaussian', 'dirichlet'.
            - ... (other parameters like num_endmembers, n_bands, beta)
    """

    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.params = params
        self.is_deep = True
        self.device = device
        n_end = params["num_endmembers"]
        n_bands = params["n_bands"]
        self.beta = params["beta"]
        self.latent_dist = params.get("latent_dist", "gaussian")

        # --- Encoder Architecture depends on the chosen latent distribution ---
        if self.latent_dist in ["gaussian", "trunc_normal"]:
            encoder_output_dim = n_end * 2
        elif self.latent_dist == "dirichlet":
            encoder_output_dim = n_end
        else:
            raise ValueError(f"Unknown latent distribution: {self.latent_dist}")

        if self.is_deep:
            self.encoder = nn.Sequential(
                nn.Linear(n_bands, n_end * 9, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 9, n_end * 6, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 6, n_end * 3, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(n_end * 3, encoder_output_dim, bias=False),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_bands, encoder_output_dim, bias=False)
            )

        # --- Utility and Decoder Layers (mostly reused) ---
        if self.latent_dist != "dirichlet":
            self.batch_norm = nn.BatchNorm1d(n_end)
            self.sparse_relu = SparseReLU(n_end)
            self.sum_to_one = SumToOneSimple()

        self.dropout = nn.Dropout(p=0.0045)
        self.decoder = nn.Linear(n_end, n_bands, bias=False)

        with torch.no_grad():
            self.decoder.weight.data.abs_()
        self.to(self.device)

    def reparameterize_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_dirichlet(self, alpha):
        # Use the Gamma distribution reparameterization trick for Dirichlet
        # This is the key to making Dirichlet sampling differentiable
        gamma_dist = Gamma(concentration=alpha, rate=1.0)
        samples = gamma_dist.rsample()  # this is where the magic happens
        # Normalize the Gamma samples to get a Dirichlet sample
        return samples / (samples.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu, logvar, abundances = None, None, None

        if self.latent_dist == "gaussian":
            mu = encoder_output[:, : self.params["num_endmembers"]]
            logvar = encoder_output[:, self.params["num_endmembers"] :]
            z = self.reparameterize_gaussian(mu, logvar)

            # Post-processing to enforce constraints
            z_norm = self.batch_norm(z)
            z_sparse = self.sparse_relu(z_norm)
            abundances = self.sum_to_one(z_sparse)

        elif self.latent_dist == "dirichlet":
            # Ensure concentration parameters alpha are positive
            # Softplus is a smooth approximation of ReLU
            alpha = F.softplus(encoder_output) + 1e-4
            abundances = self.reparameterize_dirichlet(alpha)
            # For Dirichlet, the "mu" for KLD calculation is alpha itself
            mu = alpha  # Using 'mu' variable to hold the distribution parameter

        abundances_dropped = self.dropout(abundances)
        decoded = self.decoder(abundances_dropped)

        return decoded, abundances, mu, logvar

    def fit(self, data, plot_every=0):

        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)

        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        optimizer = self.params["optimizer"]
        loss_fn = self.params["loss"]

        total_steps = len(dataloader) * self.params["epochs"]
        anneal_start_step = int(0.1 * total_steps)
        current_step = 0

        # annealing
        history = []
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                current_step += 1
                # annealing
                if current_step < anneal_start_step:
                    current_beta = 0.0
                else:
                    anneal_progress = (current_step - anneal_start_step) / (
                        total_steps - anneal_start_step
                    )
                    current_beta = self.beta * min(anneal_progress, 1.0)
                # no annealing
                # current_beta = self.beta

                optimizer.zero_grad()
                outputs, _, mu, logvar = self.forward(batch_x)
                recon_loss = loss_fn(outputs, batch_y)

                # --- KL Divergence calculation depends on the distribution ---
                if self.latent_dist == "gaussian":
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                elif self.latent_dist == "dirichlet":
                    alpha = mu  # We stored alpha in the 'mu' variable
                    P = self.params["num_endmembers"]
                    # Prior is a uniform Dirichlet with beta_0 = [1, 1, ..., 1]
                    beta_0 = torch.ones_like(alpha)

                    # Formula for D_KL(Dir(alpha) || Dir(beta_0))
                    kld_loss = (
                        torch.lgamma(alpha.sum(-1))
                        - torch.lgamma(alpha).sum(-1)
                        - torch.lgamma(beta_0.sum(-1))
                        + torch.lgamma(beta_0).sum(-1)
                        + (
                            (alpha - beta_0)
                            * (
                                torch.digamma(alpha)
                                - torch.digamma(alpha.sum(-1, keepdim=True))
                            )
                        ).sum(-1)
                    )
                    kld_loss = kld_loss.sum()

                kld_loss /= batch_x.size(0)
                total_loss = recon_loss + current_beta * kld_loss
                total_loss.backward()

                with torch.no_grad():
                    self.decoder.weight.data.clamp_(min=0)
                optimizer.step()
                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history.append(avg_loss)
            if plot_every > 0 and (epoch + 1) % plot_every == 0:
                print(
                    f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {avg_loss:.6f}, Beta: {current_beta:.4f}"
                )
        return history

    def get_abundances(self):
        # For inference, it's common to use the mean 'mu' directly
        # instead of a random sample for deterministic output.
        self.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(self.params["data"].array()).to(self.device)

            # Pass through encoder to get mu and logvar
            mu_logvar = self.encoder(data_tensor)
            mu = mu_logvar[:, : self.params["num_endmembers"]]

            # Process 'mu' as the latent code
            mu_norm = self.batch_norm(mu)
            mu_sparse = self.sparse_relu(mu_norm)
            abundances = self.sum_to_one(mu_sparse)

            abundances_np = abundances.cpu().numpy()
            abundances_reshaped = np.reshape(
                abundances_np,
                [
                    self.params["data"].cols,
                    self.params["data"].rows,
                    self.params["num_endmembers"],
                ],
            )
        self.train()
        return abundances_reshaped


############################################
