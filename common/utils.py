import copy
import glob
import json
import os
import random
import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-reward({reward:+.2E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')
CHECKPOINT_PATTERN = re.compile(r'^(.*\/|.*\\)?[\w-]*epoch\((?P<epoch>\d+)\)-reward\((?P<reward>[\-+Ee\d.]+)\)[\w-]*\.pkl$')


def clone_network(src_net, device=None):
    if device is None:
        device = getattr(src_net, 'device', None)

    dst_net = copy.deepcopy(src_net)

    if device is not None:
        dst_net.to(device)

    return dst_net


def sync_params(src_net, dst_net, soft_tau=1.0):
    assert 0.0 <= soft_tau <= 1.0
    assert type(src_net) == type(dst_net)

    if soft_tau == 0.0:
        return
    elif soft_tau == 1.0:
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(src_param.data)
    else:  # 0.0 < soft_tau < 1.0
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(dst_param.data * (1.0 - soft_tau) + src_param.data * soft_tau)


def init_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        n_params = 0
        for param in param_group['params']:
            n_params += param.size().numel()
        param_group['n_params'] = n_params


def clip_grad_norm(optimizer, max_norm=None, norm_type=2):
    for param_group in optimizer.param_groups:
        max_norm_x = max_norm
        if max_norm_x is None and 'n_params' in param_group:
            max_norm_x = 0.1 * np.sqrt(param_group['n_params'])
        if max_norm_x is not None:
            nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                               max_norm=max_norm,
                                               norm_type=norm_type)


def check_devices(config):
    if config.gpu is not None and torch.cuda.is_available():
        if len(config.gpu) == 0:
            config.gpu = [0]
        devices = []
        for device in config.gpu:
            if isinstance(device, int):
                devices.append(torch.device(f'cuda:{device}'))
            elif device in ('c', 'cpu', 'C', 'CPU'):
                devices.append(torch.device('cpu'))
    else:
        devices = [torch.device('cpu')]

    config.devices = devices

    return devices


def get_checkpoint(checkpoint_dir, by='epoch', specified_epoch=None):
    ''' get checkpoint file by epoch or reward '''
    try:
        checkpoints = glob.iglob(os.path.join(checkpoint_dir, '*.pkl'))
        matches = filter(None, map(CHECKPOINT_PATTERN.match, checkpoints))

        if specified_epoch is not None and isinstance(specified_epoch, int):
            for match in matches:
                checkpoint_epoch = int(match.group('epoch'))
                if checkpoint_epoch == specified_epoch:
                    return match.group()

        max_match = max(matches, key=lambda match: float(match.group(by)), default=None)
        return max_match.group()
    except AttributeError:
        return None


def check_logging(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    try:
        from IPython.core.formatters import PlainTextFormatter
        formatter = PlainTextFormatter()
    except ImportError:
        formatter = str

    for directory in (config.log_dir, config.checkpoint_dir):
        with open(file=os.path.join(directory, 'config.json'), mode='w') as file:
            json.dump(config, file, indent=4, default=formatter)

    if config.mode == 'test' or config.mode == 'test_render' or config.load_checkpoint:
        initial_checkpoint = get_checkpoint(config.checkpoint_dir, by='epoch')
    else:
        initial_checkpoint = None
    if initial_checkpoint is not None:
        initial_epoch = int(CHECKPOINT_PATTERN.search(initial_checkpoint).group('epoch'))
    else:
        initial_epoch = 0

    print(f'load checkpoint from {initial_checkpoint}')
    config.initial_checkpoint = initial_checkpoint
    config.initial_epoch = initial_epoch


def center_crop(img, desired_size, shift_H=1, shift_W=1):
    ''' Crop to get a new image with desired size.


        `img` is numpy array image in `H x W x C` format.
        `desired_size` is a tuple in format `(H, W)`.
        `shift_H` and `shift_W` shift center of image before center crop by specified factor.
    '''
    H, W = img.shape[:2]
    desired_H, desired_W = desired_size

    H_cen = int(H // 2 * shift_H)
    W_cen = int(W // 2 * shift_W)

    y = H_cen - desired_H // 2
    x = W_cen - desired_W // 2

    return np.ascontiguousarray(img[y:y+desired_H, x:x+desired_W])


def sample_carracing_bias_action(prev_action):
    ''' Sample bias action for CarRacing-v0 '''
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action * mask


def sample_carla_bias_action():
    ''' Sample bias action for Carla-v0 '''
    longitudinal = random.random()
    if longitudinal > 0.4:
        # 60% chance of accelerating at least 5%
        acc = min(5, random.random())
    else:
        # 40% chance of brake between 5% and 70%
        acc = -random.random()
        acc = max(-0.05, min(acc, -0.7))

    lateral = random.random()
    if lateral > 0.5:
        steer = 0
    else:
        steer = random.gauss(0, 1)

    return np.array([acc, steer], dtype=np.float32)


def _transform_np_image_to_tensor(imgs, normalize=True):
    tensors = []
    # print('imgs type', type(imgs))
    # print('imgs shape', imgs.shape)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img.transpose(2, 0, 1)
        if normalize:
            img = img / 255.

        img_tensor = torch.from_numpy(img)
        # print('img_tensor', img_tensor.size())
        # print('img_tensor type', img_tensor.dtype)
        tensors.append(img_tensor)

    return torch.stack(tensors, dim=0)

def _transform_tensor(img_tensors, normalize=True):
    tensors = []
    # print('img_tensors type', type(img_tensors))
    # print('img_tensors size', img_tensors.size())
    for i in range(img_tensors.size(0)):
        img = img_tensors[i]

        if img.size(0) == 3:
            # correct form, do nothing
            pass
        elif img.size(-1) == 3:
            img = img.permute((2, 0, 1)).contiguous()

        assert img.size(0) == 3

        if normalize:
            img = img / 255.

        # print(img_tensor)
        # print('img_tensor', img_tensor.size())
        tensors.append(img)

    return torch.stack(tensors, dim=0)


def encode_vae_observation(observation, encoder, normalize=True, device='cpu', output_device='cpu'):
    ''' transforms ``Tensor`` or numpy's ``ndarray`` to state vector.

    If observation is ``ndarray``, it must be in form of ``batch`` x ``n_images`` x ``H`` x ``W`` x ``C``.

    If observation is ``Tensor``, it must be in form of ``batch`` x ``n_images`` x ``C`` x ``H`` x ``W`` or ``batch`` x ``n_images`` x ``H`` x ``W`` x ``C``.
    
    returns ``batch`` x ``state size`` tensor on CPU.
    
    ``device`` is device that encoder will run on.'''
    if isinstance(observation, np.ndarray):
        # print('raw observation', observation.shape)
        if len(observation.shape) == 4:
            observation = np.expand_dims(observation, axis=0)
        elif len(observation.shape) != 5:
            raise ValueError(f'observation must have 4 or 5 dimensions, current input shape is {observation.shape}')

        assert observation.ndim == 5
        batch_size = observation.shape[0]
        # print('observation.shape', observation.shape)
        # observation = observation.squeeze()
        observation_tensors = [_transform_np_image_to_tensor(observation[i], normalize).to(device) for i in range(observation.shape[0])]
    elif isinstance(observation, torch.Tensor):
        assert len(observation.size()) == 5

        batch_size = observation.shape[0]

        observation_tensors = [_transform_tensor(observation[i], normalize).to(device) for i in range(observation.size(0))]

    batch_states = []
    with torch.no_grad():
        for observation_tensor in observation_tensors:
            # print('observation_tensor type', type(observation_tensor))
            # print('observation_tensor size', observation_tensor.size())
            observation_state_batch = encoder(observation_tensor, encode=True, mean=True)
            # print('observation_state_batch size', observation_state_batch.size())
            observation_state = observation_state_batch.flatten()
            # print('observation_state size', observation_state.size())
            batch_states.append(observation_state)
        
    new_state = torch.stack(batch_states, dim=0)
    # print('new_state size', new_state.size())
    assert new_state.size(0) == batch_size

    return new_state.to(output_device)
