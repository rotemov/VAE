from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def ELBO_loss(output, input):
    x, sigma, mu = output
    prediction_dim = input[0].shape[0]
    variance = sigma.pow(2)
    mu_squared = mu.pow(2)
    kld_element = 0.5 * (1 + variance.log() - variance - mu_squared)
    kld_loss = kld_element.sum()
    prediction_mu = x[:, :prediction_dim]
    prediction_variance = x[:, prediction_dim:].exp()
    reconstruction_criterion = nn.MSELoss(size_average=False)
    reconstruction_loss = reconstruction_criterion(prediction_mu, input)
    # reconstruction_loss = 1 / x.shape[0] * torch.sum(torch.divide(torch.square(input - prediction_mu), 2 * prediction_variance))
    elbo_loss = kld_loss + reconstruction_loss
    return elbo_loss


class VAE_2(nn.Module):
    CRITERION = ELBO_loss

    def __init__(self, latent_dim):
        super(VAE_2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400))
        self.logvar = nn.Sequential(nn.Linear(400, 20))
        self.mu = nn.Sequential(nn.Linear(400, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.Linear(400, 2 * 28 * 28),  # first half is mean second is variance
            nn.Tanh())
        """
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True))
        self.logvar = nn.Sequential(
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
        """

    def forward(self, x):
        x = self.encoder(x)
        # Stochastic layer
        sigma = self.logvar(x).mul(0.5).exp()
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


def loss_function(output, input):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    recon_x, mu, logvar = output
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, input)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


class VAE(nn.Module):
    CRITERION = loss_function

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def predict(self, x):
        prediction, _, _ = self.forward(x)
        return prediction
