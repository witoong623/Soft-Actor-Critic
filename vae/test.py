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

from dataloader import get_vae_dataloader
from main import test
from common.network import ConvBetaVAE


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 10

LATENT_SIZE = 512
LEARNING_RATE = 5e-4

USE_CUDA = True
PRINT_INTERVAL = 200
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './vae_comparisions/'

if __name__ == "__main__":

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())

    # train_loader = get_dataloader('/root/thesis/thesis-code/Soft-Actor-Critic/carla_images', BATCH_SIZE, 3)
    test_loader = get_vae_dataloader('/root/thesis/thesis-code/Soft-Actor-Critic/carla_test_images', BATCH_SIZE, 3)

    print('latent size:', LATENT_SIZE)
    # TODO: update size of image
    model = ConvBetaVAE((270, 480), latent_size=LATENT_SIZE)
    model.load_model('vae_weights/Carla-v0/epoch(10)-loss(+2.294E+05).pkl')
    model = model.to(device)

    test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

    save_image(original_images + rect_images, COMPARE_PATH + 'carla_model_10_epoch' + '.jpeg', padding=0, nrow=len(original_images))

    # train_losses.append((epoch, train_loss))
    # test_losses.append((epoch, test_loss))
    # utils.write_log(LOG_PATH, (train_losses, test_losses))
