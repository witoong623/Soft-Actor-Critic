import argparse
import glob
import os
import random
import re
from collections import OrderedDict
from datetime import datetime

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.environment import FlattenedAction, NormalizedAction, FlattenedObservation
from common.network_base import VanillaNeuralNetwork


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Train or test Soft Actor-Critic controller.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                    help='mode (default: train)')
parser.add_argument('--gpu', type=int, default=None, nargs='+', metavar='CUDA_DEVICE',
                    help='GPU devices (use CPU if not present)')
parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                    help='environment to train on (default: BipedalWalker-v3)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--net', type=str, choices=['FC', 'RNN'], default='FC',
                    help='architecture of controller network')
parser.add_argument('--activation', type=str, choices=['ReLU', 'LeakyReLU'], default='ReLU',
                    help='activation function in networks (default: ReLU)')
parser.add_argument('--deterministic', action='store_true', help='deterministic in evaluation')
fc_group = parser.add_argument_group('FC controller')
fc_group.add_argument('--hidden-dims', type=int, default=[512], nargs='+',
                      help='hidden dimensions of FC controller')
rnn_group = parser.add_argument_group('RNN controller')
rnn_group.add_argument('--hidden-dims-before-lstm', type=int, default=[512], nargs='+',
                       help='hidden FC dimensions before LSTM layers in RNN controller')
rnn_group.add_argument('--hidden-dims-lstm', type=int, default=[512], nargs='+',
                       help='LSTM hidden dimensions of RNN controller')
rnn_group.add_argument('--hidden-dims-after-lstm', type=int, default=[512], nargs='+',
                       help='hidden FC dimensions after LSTM layers in RNN controller')
rnn_group.add_argument('--skip-connection', action='store_true', default=False,
                       help='add skip connection beside LSTM layers in RNN controller')
rnn_group.add_argument('--step-size', type=int, default=16,
                       help='number of continuous steps for update (default: 16)')
encoder_group = parser.add_argument_group('state encoder')
encoder_group.add_argument('--state-dim', type=int, default=None,
                           help='target state dimension of encoded state (use env.observation_space.shape if not present)')
encoder_group.add_argument('--encoder-hidden-dims', type=int, default=[], nargs='+',
                           help='hidden dimensions of FC state encoder')
parser.add_argument('--max-episode-steps', type=int, default=10000,
                    help='max steps per episode (default: 10000)')
parser.add_argument('--n-epochs', type=int, default=1000,
                    help='number of learning epochs (default: 1000)')
parser.add_argument('--n-updates', type=int, default=32,
                    help='number of learning updates per epoch (default: 32)')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size (default: 256)')
parser.add_argument('--n-samplers', type=int, default=4,
                    help='number of parallel samplers (default: 4)')
parser.add_argument('--buffer-capacity', type=int, default=1000000,
                    help='capacity of replay buffer (default: 1000000)')
parser.add_argument('--update-sample-ratio', type=float, default=2.0,
                    help='speed ratio of training and sampling (default: 2.0)')
lr_group = parser.add_argument_group('learning rate')
lr_group.add_argument('--lr', type=float, default=1E-4,
                      help='learning rate (can be override by the following specific learning rate) (default: 0.0001)')
lr_group.add_argument('--soft-q-lr', type=float, default=None,
                      help='learning rate for Soft Q Networks (use LR above if not present)')
lr_group.add_argument('--policy-lr', type=float, default=None,
                      help='learning rate for Policy Networks (use LR above if not present)')
alpha_group = parser.add_argument_group('temperature parameter')
alpha_group.add_argument('--alpha-lr', type=float, default=None,
                         help='learning rate for temperature parameter (use LR above if not present)')
alpha_group.add_argument('--initial-alpha', type=float, default=1.0,
                         help='initial value of temperature parameter (default: 1.0)')
alpha_group.add_argument('--auto-entropy', action='store_true',
                         help='auto update temperature parameter while training')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--random-seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--log-dir', type=str, default=os.path.join(ROOT_DIR, 'logs'),
                    help='folder to save tensorboard logs')
parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoints'),
                    help='folder to save checkpoint from')
parser.add_argument('--load-checkpoint', action='store_true',
                    help='load latest checkpoint in checkpoint dir')
args = parser.parse_args()

MODE = args.mode

USE_LSTM = (args.net == 'RNN')
if USE_LSTM:
    HIDDEN_DIMS_BEFORE_LSTM = args.hidden_dims_before_lstm
    HIDDEN_DIMS_AFTER_LSTM = args.hidden_dims_after_lstm
    HIDDEN_DIMS_LSTM = args.hidden_dims_lstm
    SKIP_CONNECTION = args.skip_connection
    STEP_SIZE = args.step_size
else:
    HIDDEN_DIMS = args.hidden_dims

if args.activation == 'ReLU':
    ACTIVATION = F.relu
else:
    ACTIVATION = F.leaky_relu

ENV_NAME = args.env
ENV = NormalizedAction(FlattenedAction(FlattenedObservation(gym.make(ENV_NAME))))
ENV_OBSERVATION_DIM = ENV.observation_space.shape[0]
STATE_DIM = (args.state_dim or ENV_OBSERVATION_DIM)
if args.state_dim is not None or len(args.encoder_hidden_dims) > 0:
    STATE_ENCODER = VanillaNeuralNetwork(n_dims=[ENV_OBSERVATION_DIM, *args.encoder_hidden_dims, STATE_DIM],
                                         activation=ACTIVATION, output_activation=None)
else:
    STATE_ENCODER = nn.Identity()
MAX_EPISODE_STEPS = args.max_episode_steps
try:
    MAX_EPISODE_STEPS = min(MAX_EPISODE_STEPS, ENV.spec.max_episode_steps)
except AttributeError:
    pass
RENDER = args.render

N_EPOCHS = args.n_epochs
N_SAMPLERS = args.n_samplers
BUFFER_CAPACITY = args.buffer_capacity
N_UPDATES = args.n_updates
BATCH_SIZE = args.batch_size
UPDATE_SAMPLE_RATIO = args.update_sample_ratio

DETERMINISTIC = args.deterministic

LR = args.lr
SOFT_Q_LR = (args.soft_q_lr or LR)
POLICY_LR = (args.policy_lr or LR)
ALPHA_LR = (args.alpha_lr or LR)

INITIAL_ALPHA = args.initial_alpha
WEIGHT_DECAY = args.weight_decay
AUTO_ENTROPY = args.auto_entropy

if args.gpu is not None and torch.cuda.is_available():
    if len(args.gpu) == 0:
        args.gpu = [0]
    DEVICES = [torch.device(f'cuda:{cuda_device}') for cuda_device in args.gpu]
else:
    DEVICES = [torch.device('cpu')]

RANDOM_SEED = args.random_seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
ENV.seed(RANDOM_SEED)

CURRENT_TIME = datetime.now().strftime('%Y-%m-%d-%T')
LOG_DIR = os.path.join(args.log_dir, CURRENT_TIME)
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_REGEX = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)\.pkl$')
if MODE == 'test' or args.load_checkpoint:
    INITIAL_CHECKPOINT = max(glob.iglob(os.path.join(CHECKPOINT_DIR, '*.pkl')),
                             key=lambda path: int(CHECKPOINT_REGEX.search(path).group('epoch')),
                             default=None)
else:
    INITIAL_CHECKPOINT = None
if INITIAL_CHECKPOINT is not None:
    INITIAL_EPOCH = int(CHECKPOINT_REGEX.search(INITIAL_CHECKPOINT).group('epoch'))
else:
    INITIAL_EPOCH = 0


def main():
    model_kwargs = {}
    update_kwargs = {}
    initial_random_sample = True
    ratio_factor = BATCH_SIZE
    if not USE_LSTM:
        model_kwargs.update({'hidden_dims': HIDDEN_DIMS})
    else:
        model_kwargs.update({
            'hidden_dims_before_lstm': HIDDEN_DIMS_BEFORE_LSTM,
            'hidden_dims_lstm': HIDDEN_DIMS_LSTM,
            'hidden_dims_after_lstm': HIDDEN_DIMS_AFTER_LSTM,
            'skip_connection': SKIP_CONNECTION
        })

    if MODE == 'train':
        model_kwargs.update({
            'soft_q_lr': SOFT_Q_LR,
            'policy_lr': POLICY_LR,
            'alpha_lr': ALPHA_LR,
            'weight_decay': WEIGHT_DECAY
        })

        if not USE_LSTM:
            from sac.model import Trainer as Model
        else:
            from sac.rnn.model import Trainer as Model
            initial_random_sample = False
            ratio_factor *= STEP_SIZE
            update_kwargs.update({'step_size': STEP_SIZE})
    else:
        if not USE_LSTM:
            from sac.model import Tester as Model
        else:
            from sac.rnn.model import Tester as Model

    model = Model(env=ENV,
                  state_encoder=STATE_ENCODER,
                  state_dim=STATE_DIM,
                  action_dim=ENV.action_space.shape[0],
                  activation=ACTIVATION,
                  initial_alpha=INITIAL_ALPHA,
                  n_samplers=N_SAMPLERS,
                  buffer_capacity=BUFFER_CAPACITY,
                  devices=DEVICES,
                  random_seed=RANDOM_SEED,
                  **model_kwargs)

    model.print_info()

    if INITIAL_CHECKPOINT is not None:
        model.load_model(path=INITIAL_CHECKPOINT)

    if MODE == 'train' and INITIAL_EPOCH < N_EPOCHS:
        while model.replay_buffer.size < 10 * ratio_factor:
            model.env_sample(n_episodes=10,
                             max_episode_steps=MAX_EPISODE_STEPS,
                             deterministic=False,
                             random_sample=initial_random_sample,
                             render=RENDER)

        collector_process = model.async_env_sample(n_episodes=np.inf,
                                                   max_episode_steps=MAX_EPISODE_STEPS,
                                                   deterministic=False,
                                                   random_sample=False,
                                                   render=RENDER,
                                                   log_dir=LOG_DIR)

        def train():
            train_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'model'), comment='model')
            n_initial_samples = model.collector.total_steps
            global_step = 0
            for epoch in range(INITIAL_EPOCH + 1, N_EPOCHS + 1):
                soft_q_loss_list = []
                policy_loss_list = []
                alpha_list = []
                ratio_list = []
                with tqdm.trange(N_UPDATES, desc=f'Training {epoch}/{N_EPOCHS}') as pbar:
                    for i in pbar:
                        soft_q_loss, policy_loss, alpha = model.update(batch_size=BATCH_SIZE,
                                                                       normalize_rewards=True,
                                                                       auto_entropy=AUTO_ENTROPY,
                                                                       target_entropy=-1.0 * model.action_dim,
                                                                       soft_tau=0.01,
                                                                       **update_kwargs)
                        global_step += 1
                        buffer_size = model.replay_buffer.size
                        try:
                            update_sample_ratio = (ratio_factor * global_step) / (model.collector.total_steps - n_initial_samples)
                        except ZeroDivisionError:
                            update_sample_ratio = UPDATE_SAMPLE_RATIO
                        soft_q_loss_list.append(soft_q_loss)
                        policy_loss_list.append(policy_loss)
                        alpha_list.append(alpha)
                        ratio_list.append(update_sample_ratio)
                        train_writer.add_scalar(tag='train/soft_q_loss', scalar_value=soft_q_loss, global_step=global_step)
                        train_writer.add_scalar(tag='train/policy_loss', scalar_value=policy_loss, global_step=global_step)
                        train_writer.add_scalar(tag='train/temperature_parameter', scalar_value=alpha, global_step=global_step)
                        train_writer.add_scalar(tag='train/buffer_size', scalar_value=buffer_size, global_step=global_step)
                        train_writer.add_scalar(tag='train/update_sample_ratio', scalar_value=update_sample_ratio, global_step=global_step)
                        pbar.set_postfix(OrderedDict([('global_step', global_step),
                                                      ('soft_q_loss', np.mean(soft_q_loss_list)),
                                                      ('policy_loss', np.mean(policy_loss_list)),
                                                      ('update_sample_ratio', update_sample_ratio)]))
                        if update_sample_ratio < UPDATE_SAMPLE_RATIO:
                            model.collector.pause()
                        else:
                            model.collector.resume()

                train_writer.add_scalar(tag='epoch/soft_q_loss', scalar_value=np.mean(soft_q_loss_list), global_step=epoch)
                train_writer.add_scalar(tag='epoch/policy_loss', scalar_value=np.mean(policy_loss_list), global_step=epoch)
                train_writer.add_scalar(tag='epoch/temperature_parameter', scalar_value=np.mean(alpha_list), global_step=epoch)
                train_writer.add_scalar(tag='epoch/update_sample_ratio', scalar_value=np.mean(ratio_list), global_step=epoch)

                train_writer.flush()
                if epoch % 100 == 0:
                    model.save_model(path=os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.pkl'))

            train_writer.close()

        trainer_process = mp.Process(target=train)

        try:
            trainer_process.start()
            trainer_process.join()
        except:
            raise
        finally:
            collector_process.terminate()
            collector_process.join()

    elif MODE == 'test':
        test_writer = SummaryWriter(log_dir=LOG_DIR)

        model.env_sample(n_episodes=N_EPOCHS,
                         max_episode_steps=MAX_EPISODE_STEPS,
                         deterministic=DETERMINISTIC,
                         random_sample=False,
                         render=RENDER,
                         writer=test_writer)
        episode_steps = np.asanyarray(model.episode_steps)
        episode_rewards = np.asanyarray(model.episode_rewards)
        average_reward = episode_rewards / episode_steps
        test_writer.add_histogram(tag='test/cumulative_reward', values=episode_rewards)
        test_writer.add_histogram(tag='test/average_reward', values=average_reward)
        test_writer.add_histogram(tag='test/episode_steps', values=episode_steps)

        try:
            import pandas as pd
            df = pd.DataFrame({
                'Metrics': ['Cumulative Reward', 'Average Reward', 'Episode Steps'],
                'Mean': list(map(np.mean, [episode_rewards, average_reward, episode_steps])),
                'Std': list(map(np.std, [episode_rewards, average_reward, episode_steps])),
            })
            print(df.to_string(index=False))
        except ImportError:
            pass

        test_writer.close()


if __name__ == '__main__':
    try:
        main()
    except:
        raise
    finally:
        ENV.close()
