import os
import torch
from torch import nn
import torch.utils.data as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import gc as gc

# Inspired by: https://github.com/L1aoXingyu/pytorch-beginner
# Learning settings
CP_TEMPLATE = './mlp_img_ld{}/sim_autoencoder.pth'
NUM_EPOCHS = 100
BATCH_SIZE = 128

# Optimizer settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Dataset settings
SIGNAL_DIGIT = 1
TEST_SET_SIZE = 10 ** 4

# Model settings
LATENT_DIMS = [3, 6, 9]

# Creating needed directories
for ld in LATENT_DIMS:
    if not os.path.exists('./mlp_img_ld{}'.format(ld)):
        os.mkdir('./mlp_img_ld{}'.format(ld))


class Autoencoder(nn.Module):
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_dataloaders():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MNIST('./data', transform=img_transform, download=True)
    sig_mask = dataset.train_labels == SIGNAL_DIGIT
    sig_idx = [i for i in range(len(dataset)) if sig_mask[i]]
    bg_idx = [i for i in range(len(dataset)) if not sig_mask[i]]
    sig = utils.Subset(dataset, sig_idx)
    bg = utils.Subset(dataset, bg_idx)
    bg_test, bg_train = utils.random_split(bg, [TEST_SET_SIZE, len(bg) - TEST_SET_SIZE])
    train_dataloader = DataLoader(bg_train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(bg_test, batch_size=BATCH_SIZE, shuffle=False)
    sig_dataloader = DataLoader(sig, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, test_dataloader, sig_dataloader


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    ref_img = None
    if os.path.exists(CP_TEMPLATE.format(model.latent_dim)):
        model, losses, last_epoch, ref_img = load_checkpoint(CP_TEMPLATE.format(model.latent_dim))
    else:
        losses = []
        last_epoch = 0

    for epoch in range(last_epoch, num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            if ref_img:
                ref_img = deepcopy(img)
                pic = to_img(ref_img.data)
                save_image(pic, './mlp_img_ld{}/ref_image.png'.format(model.latent_dim))

            # Forward
            output = model(img)
            loss = criterion(output, img)

            # Back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        losses.append(loss.item())
        output = model(ref_img)
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img_ld{}/image_{}.png'.format(model.latent_dim, epoch))
        if epoch % 10 == 0:
            save_checkpoint(model, losses, epoch, ref_img)


def save_checkpoint(model, losses, epoch, ref_img):
    cp_dict = {
        'model_state_dict': model.state_dict,
        'losses': losses,
        'epoch': epoch,
        'ref_img': ref_img,
        'latent_dim': model.latent_dim
    }
    torch.save(cp_dict, CP_TEMPLATE.format(model.latent_dim))


def load_checkpoint(cp_path):
    cp = torch.load(cp_path)
    model = Autoencoder(cp['latent_dim'])
    model.load_state_dict(cp['model_state_dict'])
    return model, cp['losses'], cp['epoch'], cp['ref_img']


def main():
    train_dataloader, test_dataloader, sig_dataloader = get_dataloaders()
    for ld in LATENT_DIMS:
        model = Autoencoder(ld).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        train_model(model, criterion, optimizer, train_dataloader, num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
