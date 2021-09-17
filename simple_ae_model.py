import torch
from torch import nn
from copy import deepcopy
from sklearn.cluster import k_means
from kmeans_pytorch import kmeans


class Autoencoder(nn.Module):
    RECONSTRUCTION_CRITERION = nn.MSELoss()
    WARMUP = 10
    CENTROIDS = None
    N_CLUSTERS = 10

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, latent_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return x, encoded

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def predict(self, x):
        prediction, _ = self.forward(x)
        return prediction

    def get_latent_value(self, x):
        return self.encode(x)

    def mse_k_means(X, Y, latent, iteration, alpha=0.1):
        # https://discuss.pytorch.org/t/k-means-loss-calculation/22041/6
        reconstruction_loss = Autoencoder.RECONSTRUCTION_CRITERION(X, Y)
        if iteration < Autoencoder.WARMUP:
            return reconstruction_loss
        elif iteration == Autoencoder.WARMUP:
            Autoencoder.CENTROIDS = kmeans(X=latent, num_clusters=Autoencoder.N_CLUSTERS)[1].detach()[1].detach()
        k_means_loss = ((latent[:, None] - Autoencoder.CENTROIDS[1]) ** 2).sum(2).min(1)[0].mean()
        loss = reconstruction_loss + alpha * k_means_loss
        return loss

    CRITERION = mse_k_means
