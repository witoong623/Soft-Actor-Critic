import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import os
import multiprocessing
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from functools import partial
# from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader
from main_function import train, test
from common.network import ConvBetaVAE


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 100

LATENT_SIZE = 512
LEARNING_RATE = 3e-4

USE_CUDA = True
LOG_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/logs/vae-training'
MODEL_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/vae_weights/Carla-v0_town7_b3_new_tanh_mse_flip'
COMPARE_PATH = './comparisons/'

if __name__ == "__main__":

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    print('latent size:', LATENT_SIZE)

    train_loader = get_dataloader('/home/witoon/thesis/datasets/carla-town7/outskirts_manual_collect_processed', BATCH_SIZE, 4)
    test_loader = get_dataloader('/home/witoon/thesis/datasets/carla-town7/outskirts_manual_collect_eval_processed', BATCH_SIZE, 2, is_train=False)

    model = ConvBetaVAE((256, 512), latent_size=LATENT_SIZE, beta=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # reduce only 1 time, set cooldown to epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, cooldown=EPOCHS, threshold=5e-3, patience=10, verbose=True)

    log_writer = SummaryWriter(log_dir=os.path.join(LOG_PATH, 'BVAE-B3', datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, log_interval=5)
        test_loss = test(model, device, test_loader, epoch, log_interval=1)
        scheduler.step(test_loss)

        log_writer.add_scalar('Loss/Train', train_loss, epoch)
        log_writer.add_scalar('Loss/Test', test_loss, epoch)

        print_loss = test_loss

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        model.save_model(os.path.join(MODEL_PATH, 'latest.pkl'))
        if epoch % 10 == 0:
            model.save_model(os.path.join(MODEL_PATH, CHECKPOINT_FORMAT(prefix='bvae_town7_', epoch=epoch, loss=print_loss)))

    log_writer.flush()
    log_writer.close()
