import sys
sys.path.append('/root/thesis/thesis-code/Soft-Actor-Critic')

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing
import time
from functools import partial
# from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import get_dataloader
from main_function import train
from common.network import ConvVAE


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 51

LATENT_SIZE = 512
LEARNING_RATE = 1e-4

USE_CUDA = True
PRINT_INTERVAL = 200
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

if __name__ == "__main__":

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())

    train_loader = get_dataloader('/root/thesis/thesis-code/Soft-Actor-Critic/carla_town7_images/outskirts', BATCH_SIZE, 3)
    # test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    print('latent size:', LATENT_SIZE)
    model = ConvVAE((270, 480), latent_size=LATENT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True)

    for epoch in range(0, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        # scheduler.step(train_loss)

        # test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        # train_losses.append((epoch, train_loss))
        # test_losses.append((epoch, test_loss))
        # utils.write_log(LOG_PATH, (train_losses, test_losses))

        model.save_model('/root/thesis/thesis-code/Soft-Actor-Critic/vae_weights/Carla-v0/latest.pkl')

        if epoch % 10 == 0:
            print(f'save weight at epoch {epoch}')
            model.save_model(os.path.join('/root/thesis/thesis-code/Soft-Actor-Critic/vae_weights/Carla-v0', CHECKPOINT_FORMAT(prefix='bvae_town7_', epoch=epoch, loss=train_loss)))
