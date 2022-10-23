import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda.amp as amp

from datetime import datetime
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_vae_dataloader, CarlaAITDataset
from common.network import ConvBetaVAE
from train_utils import TrainEpoch, BVAELoss


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 8
TEST_BATCH_SIZE = 256
EPOCHS = 100

LATENT_SIZE = 512
LEARNING_RATE = 3e-4

USE_CUDA = True
LOG_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/logs/carla-ait-vae'
MODEL_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/vae_weights/carla-ait-2-img'
COMPARE_PATH = './comparisons/'


def train_loop():
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    train_dataset = CarlaAITDataset()

    train_loader = data.DataLoader(train_dataset, BATCH_SIZE, num_workers=0)

    model = ConvBetaVAE((256, 512), latent_size=LATENT_SIZE, beta=3).to(device)

    loss = BVAELoss()

    grad_scaler = amp.GradScaler()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_epoch = TrainEpoch(model, loss, optimizer, grad_scaler, device)

    log_writer = SummaryWriter(log_dir=os.path.join(LOG_PATH, 'BVAE-B3', datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

    for epoch in range(1, EPOCHS + 1):
        train_log = train_epoch.run(train_loader)

        log_writer.add_scalar('Loss/Train', train_log['BetaVAELoss'], epoch)

        print_loss = train_log['BetaVAELoss']

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        model.save_model(os.path.join(MODEL_PATH, 'latest.pkl'))
        if epoch % 10 == 0:
            model.save_model(os.path.join(MODEL_PATH, CHECKPOINT_FORMAT(prefix='carla-ait-2-img', epoch=epoch, loss=print_loss)))

    log_writer.flush()
    log_writer.close()


if __name__ == "__main__":
    train_loop()
