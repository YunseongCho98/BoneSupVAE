import torch
import torch.nn as nn

def loss_fn(x_recon, x, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Kullback-Leibler divergence
    return recon + beta * kld