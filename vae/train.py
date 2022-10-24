import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.cuda.amp as amp

from datetime import datetime
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import get_vae_dataloader, CarlaAITDataset
from models import CarlaConvVAE
from train_utils import TrainEpoch, BetaVAELoss


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-loss({loss:+.3E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')

# parameters
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
EPOCHS = 100

LATENT_SIZE = 512
LEARNING_RATE = 3e-4

USE_CUDA = True
LOG_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/logs/carla-ait-vae'
MODEL_PATH = '/home/witoon/thesis/code/Soft-Actor-Critic/vae_weights/carla-ait-2-img'
COMPARE_PATH = './comparisons/'

def init_DDP():
    dist.init_process_group(backend="nccl")


def train_loop():
    train_dataset = CarlaAITDataset()

    train_sampler = data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = data.DataLoader(train_dataset, BATCH_SIZE, num_workers=2, sampler=train_sampler, pin_memory=True)

    model = CarlaConvVAE((256, 512), 6, [64, 128, 256], latent_size=768, activation=nn.ELU())
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(LOCAL_RANK)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    loss = BetaVAELoss()

    grad_scaler = amp.GradScaler()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_epoch = TrainEpoch(model, loss, optimizer, grad_scaler, LOCAL_RANK)

    log_writer = SummaryWriter(log_dir=os.path.join(LOG_PATH, 'BVAE-B3', datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

    for epoch in range(1, EPOCHS + 1):
        train_log = train_epoch.run(train_loader)

        log_writer.add_scalar('Loss/Train', train_log['BetaVAELoss'].item(), epoch)

        print_loss = train_log['BetaVAELoss']

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        if LOCAL_RANK in [-1, 0]:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'latest.pkl'))
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, CHECKPOINT_FORMAT(prefix='carla-ait-2-img', epoch=epoch, loss=print_loss)))

    log_writer.flush()
    log_writer.close()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    init_DDP()

    train_loop()
