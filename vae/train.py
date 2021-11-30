import sys
sys.path.append('/root/thesis/thesis-code/Soft-Actor-Critic')

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing
import time
from functools import partial
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import get_train_dataloader
from vae.model import ConvVAE


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    t = tqdm(train_loader)

    for batch_idx, data in enumerate(t):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item()

        if batch_idx % log_interval == 0:
            t.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 512
TEST_BATCH_SIZE = 10
EPOCHS = 400

LATENT_SIZE = 128
LEARNING_RATE = 5e-4

USE_CUDA = True
PRINT_INTERVAL = 100
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

if __name__ == "__main__":

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())

    train_loader = get_train_dataloader('/root/image_dataset', BATCH_SIZE, 2)
    # test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    print('latent size:', LATENT_SIZE)
    model = ConvVAE((96, 96), latent_size=LATENT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    for epoch in range(0, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        optimizer.step(train_loss)

        # test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        # train_losses.append((epoch, train_loss))
        # test_losses.append((epoch, test_loss))
        # utils.write_log(LOG_PATH, (train_losses, test_losses))

        if epoch % 10 == 0:
            print(f'save weight at epoch {epoch}')
            model.save_model(os.path.join('/root/thesis/thesis-code/Soft-Actor-Critic/vae_weights', CHECKPOINT_FORMAT(epoch=epoch, loss=train_loss)))
