from torch import nn
import torch


def ELBO_loss(output, input):
    x, sigma, mu = output
    prediction_dim = int(x[0].shape[0] / 2)
    variance = torch.square(sigma)
    kld_loss = 0.5 * torch.sum(1 + torch.log(variance) + torch.square(mu) + variance)
    prediction_mu = x[:, :prediction_dim]
    prediction_variance = torch.exp(x[:, prediction_dim:])
    reconstruction_loss = 1 / x.shape[0] * torch.sum(
        torch.divide(torch.square(input - prediction_mu), 2 * prediction_variance))
    elbo_loss = kld_loss + reconstruction_loss
    return elbo_loss


class VAE(nn.Module):
    CRITERION = ELBO_loss

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True))
        self.sigma = nn.Sequential(
            nn.Linear(12, int(latent_dim)),
            nn.Tanh())
        self.mu = nn.Sequential(
            nn.Linear(12, int(latent_dim)),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim), 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 2 * 28 * 28),  # first half is mean second is variance
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        # Stochastic layer
        sigma = torch.exp(self.sigma(x))
        mu = self.mu(x)
        eps = torch.randn_like(sigma)
        z = sigma * eps + mu
        x = self.decoder(z)
        return x, sigma, mu

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def predict(self, x):
        x, _, _ = self.forward(x)
        prediction_dim = int(x[0].shape[0] / 2)
        return x[:, :prediction_dim]
