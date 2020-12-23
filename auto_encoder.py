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
from matplotlib import pyplot as plt
import numpy as np

# Inspired by: https://github.com/L1aoXingyu/pytorch-beginner
# Learning settings
CP_TEMPLATE = './mlp_img_ld{}/sim_autoencoder.pth'
NUM_EPOCHS = 100
BATCH_SIZE = 128

# Optimizer settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Dataset settings
SIGNAL_DIGIT = 5
GENERATE_DIGIT = 8
TEST_SET_SIZE = 10 ** 4

# Model settings
LATENT_DIMS = [2, 3, 6, 9, 12]

# Plot settings
NBINS = 30

# Creating needed directories
for ld in LATENT_DIMS:
    if not os.path.exists('./mlp_img_ld{}'.format(ld)):
        os.mkdir('./mlp_img_ld{}'.format(ld))
    if not os.path.exists(os.path.join('./mlp_img_ld{}'.format(ld), "images")):
        os.mkdir(os.path.join('./mlp_img_ld{}'.format(ld), "images"))


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

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


def get_dataloaders():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MNIST('./data', transform=img_transform, download=True)
    sig_mask = dataset.train_labels == SIGNAL_DIGIT
    gen_mask = dataset.train_labels == GENERATE_DIGIT
    sig_idx = [i for i in range(len(dataset)) if sig_mask[i]]
    bg_idx = [i for i in range(len(dataset)) if not sig_mask[i]]
    sig = utils.Subset(dataset, sig_idx)
    bg = utils.Subset(dataset, bg_idx)
    bg_test, bg_train = utils.random_split(bg, [TEST_SET_SIZE, len(bg) - TEST_SET_SIZE])
    gen_idx = [i for i in range(len(bg_train)) if bg_train[i][1] == GENERATE_DIGIT]
    gen = utils.Subset(bg_train, gen_idx)
    data_loaders = {
        "Training data": DataLoader(bg_train, batch_size=BATCH_SIZE, shuffle=True),
        "Test data": DataLoader(bg_test, batch_size=BATCH_SIZE, shuffle=False),
        "Anomalies": DataLoader(sig, batch_size=BATCH_SIZE, shuffle=False),
        "Generate": DataLoader(gen, batch_size=BATCH_SIZE, shuffle=False)
    }
    return data_loaders


def to_pic(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def data_to_img(data):
    img, _ = data
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    return img


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    ref_img = None
    if os.path.exists(CP_TEMPLATE.format(model.latent_dim)):
        model, optimizer, losses, last_epoch, ref_img = load_checkpoint(CP_TEMPLATE.format(model.latent_dim),
                                                                        model, optimizer)
    else:
        losses = []
        last_epoch = 0

    for epoch in range(last_epoch, num_epochs):
        loss_sum = 0
        for data in dataloader:
            img = data_to_img(data)
            if ref_img is None:
                ref_img = deepcopy(img)
                pic = to_pic(ref_img.data)
                save_image(pic, './mlp_img_ld{}/images/ref_image.png'.format(model.latent_dim))

            # Forward
            output = model(img)
            loss = criterion(output, img)

            # Back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            gc.collect()

        loss_avg = loss_sum / len(dataloader)
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss_avg))
        losses.append(loss_avg)
        output = model(ref_img)
        pic = to_pic(output.cpu().data)
        save_image(pic, './mlp_img_ld{}/images/image_{}.png'.format(model.latent_dim, epoch))
        save_checkpoint(model, losses, epoch, ref_img)
        gc.collect()


def plot_training_loss(losses, result_dir):
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Training: Avg. Loss")
    plt.ylabel("log(E[Loss])")
    plt.xlabel("Epoch #")
    plt.savefig(os.path.join(result_dir, "training_loss.png"))
    plt.close()


def get_batch_losses(model, dataloader, criterion, result_dir, dset_name, input_noise=False, latent_noise=False):
    losses = [0 for i in range(len(dataloader))]
    noise_losses = deepcopy(losses)
    for i, data in enumerate(dataloader):
        img = data_to_img(data)
        output = model(img)
        loss = criterion(output, img)
        losses[i] = loss.item()
        if input_noise:  # denoising digits
            noised_img = img + (torch.randn_like(img) * 0.5)  # Adding noise with mean 0 and 0.5 std which is S/N = 1
            noised_output = model(noised_img)
        elif latent_noise:  # generating digits
            noised_img = model.encode(img)
            noised_img += torch.randn_like(noised_img)
            noised_output = model.decode(noised_img)
        if input_noise or latent_noise:
            noised_loss = criterion(noised_output, img)
            noise_losses[i] = noised_loss.item()
    if input_noise or latent_noise:
        save_image(to_pic(noised_output), os.path.join(result_dir, "noised_out_{}.png".format(dset_name)))
    if input_noise:
        save_image(to_pic(noised_img), os.path.join(result_dir, "noised_ref_{}.png".format(dset_name)))
    save_image(to_pic(img), os.path.join(result_dir, "ref_{}.png".format(dset_name)))
    save_image(to_pic(output), os.path.join(result_dir, "out_{}.png".format(dset_name)))
    return losses, noise_losses


def plot_batch_loss_histograms(model, dataloaders, criterion, result_dir):
    datasets = dataloaders.keys()
    traditional = 1
    noise = 2
    for dset in datasets:
        print("Performing {} tests".format(dset))
        input_noise_flag = dset == "Test data"
        latent_noise_flag = dset == "Generate"
        losses, noise_losses = get_batch_losses(model, dataloaders[dset], criterion, result_dir, dset, input_noise_flag,
                                                latent_noise_flag)
        plt.figure(traditional)
        if not latent_noise_flag:
            plt.hist(np.array(losses), bins=NBINS, label=dset)
        plt.figure(noise)
        if input_noise_flag:
            plt.hist(np.array(losses), bins=NBINS, label=dset)
            plt.hist(np.array(noise_losses), bins=NBINS, label="Noised {}".format(dset))
        elif latent_noise_flag:
            plt.hist(np.array(losses), bins=NBINS, label=dset)
            plt.hist(np.array(noise_losses), bins=NBINS, label="Generated {}".format(GENERATE_DIGIT))
    file_names = {
        traditional: "loss_histograms.png",
        noise: "loss_histograms_noise.png"
    }
    for fig in [traditional, noise]:
        plt.figure(fig)
        plt.yscale("log")
        plt.title("Dataset loss distributions")
        plt.xlabel("Loss")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(os.path.join(result_dir, file_names[fig]))
        plt.close()


def evaluate_model(model, criterion, optimizer, dataloaders, cp_path):
    model, optimizer, losses, _, _ = load_checkpoint(cp_path, model, optimizer)
    model.eval()
    result_dir = './mlp_img_ld{}'.format(model.latent_dim)
    plot_training_loss(losses, result_dir)
    plot_batch_loss_histograms(model, dataloaders, criterion, result_dir)


def save_checkpoint(model, losses, epoch, ref_img):
    cp_dict = {
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'epoch': epoch,
        'ref_img': ref_img,
        'latent_dim': model.latent_dim
    }
    torch.save(cp_dict, CP_TEMPLATE.format(model.latent_dim))


def load_checkpoint(cp_path, model, optimizer):
    cp = torch.load(cp_path)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.params = model.parameters()
    return model, optimizer, cp['losses'], cp['epoch'], cp['ref_img']


def main():
    print("Starting run")
    data_loaders = get_dataloaders()
    print("Datasets loaded for signal digit {} and generate digit {}".format(SIGNAL_DIGIT, GENERATE_DIGIT))
    for ld in LATENT_DIMS:
        print("Staring run for latent dim: {}".format(ld))
        model = Autoencoder(ld).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print("Starting training learning rate {} and weight decay {}".format(LEARNING_RATE, WEIGHT_DECAY))
        # train_model(model, criterion, optimizer, data_loaders["Training data"], num_epochs=NUM_EPOCHS)
        print("Starting tests")
        evaluate_model(model, criterion, optimizer, data_loaders, cp_path=CP_TEMPLATE.format(ld))


if __name__ == "__main__":
    main()
