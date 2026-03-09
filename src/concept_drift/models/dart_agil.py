from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DARTAGILNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return logits, mu, logvar, z


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def adversarial_perturbation(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        logits, _, _, _ = model(x + delta)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        delta.data = (delta + eps * delta.grad.detach().sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return delta.detach()
