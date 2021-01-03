from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def ELBO_loss_function(output, input):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    recon_x, mu, logvar = output
    batch_size = input.shape[0]
    reconstruction_function = nn.MSELoss(size_average=False)
    MSE = reconstruction_function(recon_x, input)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD


class VAE(nn.Module):
    CRITERION = ELBO_loss_function

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400),
                                     nn.ReLU(),
                                     nn.Linear(400, 128),
                                     nn.ReLU())
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 400),
                                     nn.ReLU(),
                                     nn.Linear(400, 784),
                                     nn.Sigmoid())
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = self.encoder(x)
        mu = self.mu(h1)
        logvar = self.logvar(h1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def predict(self, x):
        prediction, _, _ = self.forward(x)
        return prediction

    def get_latent_value(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
