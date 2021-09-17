from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gc as gc
from tqdm import tqdm


def ELBO_loss_function(beta=0.5):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    def loss_function(output, input):
        recon_x, mu, logvar = output
        batch_size = input.shape[0]
        reconstruction_function = nn.MSELoss(size_average=False)
        MSE = reconstruction_function(recon_x, input)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return (1 - beta)* MSE + beta * KLD
    return loss_function


class VAE(nn.Module):
    CRITERION = ELBO_loss_function()
    
    def __init__(self, latent_dim, input_dim=28*28, beta=0.5):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                     nn.ReLU(),
                                     nn.Linear(input_dim // 2, input_dim // 4),
                                     nn.ReLU())
        self.mu = nn.Linear(input_dim // 4, latent_dim)
        self.logvar = nn.Linear(input_dim // 4, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim // 4),
                                     nn.ReLU(),
                                     nn.Linear(input_dim // 4, input_dim // 2),
                                     nn.ReLU(),
                                     nn.Linear(input_dim // 2, input_dim),
                                     nn.Sigmoid())
        self.criterion = ELBO_loss_function(beta)

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
    
    def train(self, dataloader, optimizer, epochs=10):
        losses = []
        for epoch in tqdm(range(epochs)):
            for X in tqdm(dataloader):
                # Forward
                output = self.forward(X)
                loss = self.criterion(output, X)
                # Back-prop
                if not loss.isinf():
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #clip gradients
                    optimizer.step()
                    losses.append(loss.item())
                    print(loss.item())
                    gc.collect()
        return losses
