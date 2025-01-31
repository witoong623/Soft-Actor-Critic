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
from collections import deque


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


def remove_policy_weight(state_dict):
    keys_to_delete = []
    for k in state_dict.keys():
        if not k.startswith('state_encoder'):
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del state_dict[k]

    return state_dict


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

    if config.sampler_gpu is not None and torch.cuda.is_available():
        if len(config.sampler_gpu) == 0:
            config.sampler_gpu = [0]
        sampler_devices = []
        for device in config.sampler_gpu:
            if isinstance(device, int):
                sampler_devices.append(torch.device(f'cuda:{device}'))
            elif device in ('c', 'cpu', 'C', 'CPU'):
                sampler_devices.append(torch.device('cpu'))
    else:
        sampler_devices = devices

    config.sampler_devices = sampler_devices

    return devices, sampler_devices


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
        initial_checkpoint = get_checkpoint(config.checkpoint_dir, by='epoch', specified_epoch=config.epoch_number)
    else:
        initial_checkpoint = None
    if initial_checkpoint is not None:
        initial_epoch = int(CHECKPOINT_PATTERN.search(initial_checkpoint).group('epoch'))
    else:
        initial_epoch = 0

    config.initial_checkpoint = initial_checkpoint
    config.initial_epoch = initial_epoch


def check_pretrained_checkpoint(config):
    if config.initial_checkpoint is None and config.pretrained_dir:
        config.pretrained_checkpoint = get_checkpoint(config.pretrained_dir, by='epoch', specified_epoch=config.epoch_number)
    else:
        config.pretrained_checkpoint = None


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


ROAD = (128, 64, 128)
ROAD_LINE = (157, 234, 50)
SIDE_WALK = (244, 35, 232)
VEHICLE = (0, 0, 142)
HUMAN = (220, 20, 60)
OTHER = (55, 90, 80)


class Label:
    ROAD = 0
    LANE_MARKING = 1
    VEHICLE = 2
    HUMAN = 3
    OTHER = 4


def convert_to_simplified_cityscape(img):
    ''' Only work with RGB image '''
    other_elements = (img[...,] != ROAD).all(axis=2) & \
                     (img[...,] != ROAD_LINE).all(axis=2) & \
                     (img[...,] != SIDE_WALK).all(axis=2) & \
                     (img[...,] != VEHICLE).all(axis=2)
    img[other_elements] = OTHER

    return img


def label_mask_to_color_mask(label_mask):
    color_mask = np.full(label_mask.shape + (3,), OTHER, dtype=np.uint8)

    color_mask[label_mask == Label.ROAD] = ROAD
    color_mask[label_mask == Label.LANE_MARKING] = ROAD_LINE
    color_mask[label_mask == Label.VEHICLE] = VEHICLE
    color_mask[label_mask == Label.HUMAN] = HUMAN

    return color_mask


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


class ObservationStacker:
    def __init__(self, n_frames, stack_axis=2):
        self.n_frames = n_frames
        self.stack_axis = stack_axis
        self.frames_queue = deque(maxlen=self.n_frames)

    def get_new_observation(self, current_observation):
        while len(self.frames_queue) < self.n_frames:
            self.frames_queue.append(current_observation)

        self.frames_queue.append(current_observation)

        return np.concatenate(self.frames_queue, axis=self.stack_axis)


def normalize_image(image, mean, std):
    ''' normalize image in numpy format by divide it by 255
        and standardize it by mean and std.
    '''
    image = image / 255.

    return (image - mean) / std


def normalize_grayscale_image(image):
    ''' normalize image in numpy format by divide it by 255
    '''
    image = image / 255.

    return image


def batch_normalize_images(images, mean, std):
    batch_size = len(images)
    stacked_imgs = np.vstack(images)
    normalized_stacked_imgs = ((stacked_imgs / 255.) - mean) / std

    normalized_stacked_imgs = normalized_stacked_imgs.transpose((2, 0, 1))

    return np.split(normalized_stacked_imgs, batch_size, axis=1)


def batch_normalize_grayscale_images(images):
    batch_size = len(images)
    stacked_imgs = np.concatenate(images, axis=1)
    normalized_stacked_imgs = stacked_imgs / 255.

    return np.split(normalized_stacked_imgs, batch_size, axis=1)


def _transform_np_image_to_tensor(imgs, normalize=True):
    # input should be (number of image, H, W, C)
    new_images = imgs.transpose(0, 3, 1, 2)

    if normalize:
        new_images = new_images / 255.

    return torch.tensor(new_images, dtype=torch.float32)


def _transform_tensor(img_tensors, normalize=True):
    # input should be (number of image, H, W, C) or (number of image, C, H, W)
    if img_tensors.size(1) == 3:
        pass
    elif img_tensors.size(-1) == 3:
        new_img_tensors = img_tensors.permute((0, 3, 1, 2)).contiguous()

    if normalize:
        new_img_tensors = new_img_tensors / 255.

    return new_img_tensors


def convert_to_CHW_tensor(tensor: torch.Tensor):
    if tensor.ndim == 4:
        return tensor.permute((0, 3, 1, 2))
    elif tensor.ndim == 3:
        return tensor.permute((2, 0, 1))
    else:
        raise ValueError(f'Only support image tensor of 3 or 4 dimensions. Received {tensor.ndim} tensor')


def encode_vae_observation(observation, encoder, normalize=True, device='cpu', output_device='cpu'):
    ''' transforms ``Tensor`` or numpy's ``ndarray`` to state vector.

    If observation is ``ndarray``, it must be in form of ``batch`` x ``n_images`` x ``H`` x ``W`` x ``C``.

    If observation is ``Tensor``, it must be in form of ``batch`` x ``n_images`` x ``C`` x ``H`` x ``W`` or ``batch`` x ``n_images`` x ``H`` x ``W`` x ``C``.
    
    returns ``batch`` x ``state size`` tensor on CPU.
    
    ``device`` is device that encoder will run on.'''
    if isinstance(observation, np.ndarray):
        observation_tensors = _transform_np_image_to_tensor(observation, normalize=normalize).to(device)
    elif isinstance(observation, torch.Tensor):
        observation_tensors = _transform_tensor(observation, normalize=normalize).to(device)
    else:
        raise TypeError(f'Unsupported type {type(observation)}')

    with torch.no_grad():
        observation_state_batch = encoder(observation_tensors, encode=True, mean=True)

    return observation_state_batch.to(output_device)
