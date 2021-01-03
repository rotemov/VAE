import os
import torch
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
from simple_ae_model import Autoencoder
from vae_model import *
import imageio

# Inspired by: https://github.com/L1aoXingyu/pytorch-beginner
# Training settings
CP_TEMPLATE = './mlp_img_ld{}/sim_autoencoder.pth'
NUM_EPOCHS = 100
BATCH_SIZE = 128

# Optimizer settings
LEARNING_RATE = 1e-3

# Dataset settings
SIGNAL_DIGIT = 5
GENERATE_DIGIT = 6
TEST_SET_SIZE = 10 ** 4

# Model settings
MODEL = VAE
LATENT_DIMS = [2]

# Test settings
NBINS = 15
LATENT_SPACE_SAMPLE_SIZE = 128


# Creating needed directories
for ld in LATENT_DIMS:
    if not os.path.exists('./mlp_img_ld{}'.format(ld)):
        os.mkdir('./mlp_img_ld{}'.format(ld))
    if not os.path.exists(os.path.join('./mlp_img_ld{}'.format(ld), "images")):
        os.mkdir(os.path.join('./mlp_img_ld{}'.format(ld), "images"))


def get_dataloaders():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MNIST('./data', transform=img_transform, download=True)
    digit_dataloaders = []
    for i in range(10):
        dig_mask = dataset.train_labels == i
        dig_idx = [i for i in range(len(dataset)) if dig_mask[i]]
        digit_ds = utils.Subset(dataset, dig_idx)
        digit_dataloaders.append(DataLoader(digit_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True))

    sig_mask = dataset.train_labels == SIGNAL_DIGIT
    bg_idx = [i for i in range(len(dataset)) if not sig_mask[i]]
    bg = utils.Subset(dataset, bg_idx)
    bg_test, bg_train = utils.random_split(bg, [TEST_SET_SIZE, len(bg) - TEST_SET_SIZE])
    gen_idx = [i for i in range(len(bg_train)) if bg_train[i][1] == GENERATE_DIGIT]
    gen = utils.Subset(bg_train, gen_idx)
    data_loaders = {
        "Training data": DataLoader(bg_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        "Test data": DataLoader(bg_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True),
        "Anomalies": digit_dataloaders[SIGNAL_DIGIT],
        "Pre latent noise": DataLoader(gen, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    }
    return data_loaders, digit_dataloaders


def to_pic(x):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def generate_gif(result_dir):
    images = []
    for i in range(NUM_EPOCHS):
        image_path = os.path.join(result_dir, "images", "image_{}.png".format(i))
        images.append(imageio.imread(image_path))
    imageio.mimsave(os.path.join(result_dir, "training.gif"), images, duration=0.1)


def data_to_img(data):
    img, _ = data
    img = img.view(img.size(0), -1)
    img = Variable(img)
    return img


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    ref_img = None
    if os.path.exists(CP_TEMPLATE.format(model.latent_dim)):
        model, optimizer, losses, last_epoch, ref_img = load_checkpoint(CP_TEMPLATE.format(model.latent_dim),
                                                                        model, optimizer)
    else:
        losses = []
        last_epoch = 0

    for epoch in range(last_epoch+1, num_epochs):
        loss_sum = 0
        for data in dataloader:
            img = data_to_img(data)
            if ref_img is None:
                ref_img = deepcopy(img)
                pic = to_pic(ref_img.data)
                save_image(pic, './mlp_img_ld{}/images/ref_image.png'.format(model.latent_dim))

            # Forward
            prediction = model(img)
            loss = criterion(prediction, img)

            # Back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            gc.collect()

        loss_avg = loss_sum / len(dataloader)
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss_avg))
        losses.append(loss_avg)
        prediction = model.predict(ref_img)
        pic = to_pic(prediction.cpu().data)
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
        output = model.predict(img)
        loss = criterion(output, img)
        losses[i] = loss.item()
        if input_noise:  # denoising digits
            noised_img = img + torch.randn_like(img) * 0.5  # Adding noise with mean 0 and 0.5 std which is S/N = 1
            noised_output = model.predict(noised_img)
        elif latent_noise:  # generating digits
            noised_img = model.get_latent_value(img)
            noised_img += torch.randn_like(noised_img) * 0.5
            noised_output = model.decode(noised_img)
        if input_noise or latent_noise:
            noised_loss = criterion(noised_output, img)
            noise_losses[i] = noised_loss.item()
    if input_noise or latent_noise:
        save_image(to_pic(noised_output), os.path.join(result_dir, "noised_out_{}.png".format(dset_name)))
    if input_noise:
        save_image(to_pic(noised_img), os.path.join(result_dir, "noised_ref_{}.png".format(dset_name)))
    save_image(to_pic(img), os.path.join(result_dir, "ref_{}.png".format(dset_name)))
    output = model.predict(img)
    save_image(to_pic(output), os.path.join(result_dir, "out_{}.png".format(dset_name)))
    return losses, noise_losses


def plot_batch_loss_histograms(model, dataloaders, criterion, result_dir):
    datasets = dataloaders.keys()
    traditional = 1
    noise = 2
    for dset in datasets:
        print("Performing {} tests".format(dset))
        input_noise_flag = dset == "Test data"
        latent_noise_flag = dset == "Pre latent noise"
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
            plt.hist(np.array(noise_losses), bins=NBINS, label="Latent noise {}".format(GENERATE_DIGIT))
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


def sample_test(model, result_dir, sample_size=BATCH_SIZE):
    z = torch.randn(sample_size, model.latent_dim)
    sample = model.decode(z)
    save_image(to_pic(sample), os.path.join(result_dir, "sample.png"))


def latent_dim_map(model, digit_dataloaders, result_dir):
    plt.figure()
    plt.title("Latent space avg. over batches")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    for i in range(len(digit_dataloaders)):
        avg_z = []
        for x in digit_dataloaders[i]:
            z = model.get_latent_value(data_to_img(x))
            z = z.detach().numpy().mean(axis=0)
            avg_z.append(z)
        avg_z = np.array(avg_z).T
        plt.scatter(avg_z[0], avg_z[1], label=str(i))
    plt.legend()
    plt.savefig(os.path.join(result_dir, "latent_space.png"))
    plt.close()
    x = np.linspace(-2, 2, 8)
    y = np.linspace(-2, 2, 8)[::-1]
    grid = torch.tensor(np.dstack(np.meshgrid(x, y)).reshape(8**2, 2)).to(torch.float)
    sample = model.decode(grid)
    save_image(to_pic(sample), os.path.join(result_dir, "grid.png"))


def evaluate_model(model, criterion, optimizer, dataloaders, digit_dataloaders, cp_path):
    model, optimizer, losses, _, _ = load_checkpoint(cp_path, model, optimizer)
    model.eval()
    model.requires_grad_(False)
    result_dir = './mlp_img_ld{}'.format(model.latent_dim)
    if model.latent_dim == 2:
        print("Creating latent dimension map")
        latent_dim_map(model, digit_dataloaders, result_dir)
    plot_training_loss(losses, result_dir)
    plot_batch_loss_histograms(model, dataloaders, criterion, result_dir)
    generate_gif(result_dir)
    print("Sampling latent dim")
    sample_test(model, result_dir, sample_size=BATCH_SIZE)


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
    data_loaders, digit_dataloaders = get_dataloaders()
    print("Datasets loaded for signal digit {} and generate digit {}".format(SIGNAL_DIGIT, GENERATE_DIGIT))
    for ld in LATENT_DIMS:
        print("Staring run for latent dim: {}".format(ld))
        model = MODEL(ld)  #.cuda()
        criterion = MODEL.CRITERION
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Starting training learning rate {}".format(LEARNING_RATE))
        train_model(model, criterion, optimizer, data_loaders["Training data"], num_epochs=NUM_EPOCHS)
        print("Starting tests")
        criterion = nn.MSELoss()
        evaluate_model(model, criterion, optimizer, data_loaders, digit_dataloaders, cp_path=CP_TEMPLATE.format(ld))


if __name__ == "__main__":
    main()
