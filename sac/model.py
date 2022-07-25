import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp

from common.collector import Collector
from common.network import VAEBase
from common.networkbase import Container
from common.utils import clone_network, sync_params, init_optimizer, \
    clip_grad_norm, encode_vae_observation
from .network import StateEncoderWrapper, SeparatedStateEncoderWrapper, Actor, Critic


__all__ = ['build_model', 'Trainer', 'Tester']


def build_model(config):
    model_kwargs = config.build_from_keys(['env_func',
                                           'env_kwargs',
                                           'state_encoder',
                                           'state_dim',
                                           'action_dim',
                                           'hidden_dims',
                                           'activation',
                                           'n_past_actions',
                                           'n_bootstrap_step',
                                           'initial_alpha',
                                           'n_samplers',
                                           'buffer_capacity',
                                           'batch_size',
                                           'n_frames',
                                           'devices',
                                           'sampler_devices',
                                           'separate_encoder',
                                           'random_seed'])
    if config.mode == 'train':
        model_kwargs.update(config.build_from_keys(['critic_lr',
                                                    'actor_lr',
                                                    'alpha_lr',
                                                    'weight_decay',
                                                    'actor_update_frequency',
                                                    'use_popart',
                                                    'beta']))

        if not config.RNN_encoder:
            Model = Trainer
        else:
            from .rnn.model import Trainer as Model
    elif config.mode == 'test_render':
        Model = RenderTester
        model_kwargs.pop('n_bootstrap_step', None)
        model_kwargs.pop('separate_encoder', None)
        model_kwargs.pop('batch_size', None)
        model_kwargs.pop('n_frames', None)
    else:
        if not config.RNN_encoder:
            Model = Tester
        else:
            from .rnn.model import Tester as Model

    model = Model(**model_kwargs)
    model.print_info()
    for directory in (config.log_dir, config.checkpoint_dir):
        with open(file=os.path.join(directory, 'info.txt'), mode='w') as file:
            model.print_info(file=file)

    if config.pretrained_checkpoint is not None:
        print(f'load pretrained checkpoint from {config.pretrained_checkpoint}')
        model.load_model(path=config.pretrained_checkpoint,
                         strict=True,
                         load_weight_only=True)

    if config.initial_checkpoint is not None:
        print(f'load checkpoint from {config.initial_checkpoint}')
        load_ret = model.load_model(path=config.initial_checkpoint,
                                    strict=config.mode != 'test_render',
                                    load_weight_only=False)

        print(f'Missing keys: {load_ret.missing_keys}')
        print(f'Unexpected keys: {load_ret.unexpected_keys}')

    return model


class ModelBase(object):
    STATE_ENCODER_WRAPPER = StateEncoderWrapper
    COLLECTOR = Collector

    def __init__(self, env_func, env_kwargs, state_encoder,
                 state_dim, action_dim, hidden_dims, activation, n_past_actions, n_bootstrap_step,
                 initial_alpha, use_popart, beta, n_samplers, buffer_capacity, batch_size, n_frames,
                 devices, sampler_devices, separate_encoder, random_seed=0):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)
        self.sampler_devices = sampler_devices

        self.n_past_actions = n_past_actions
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.separate_encoder = separate_encoder
        if self.separate_encoder:
            self.state_encoder = SeparatedStateEncoderWrapper(state_encoder)
        else:
            self.state_encoder = self.STATE_ENCODER_WRAPPER(state_encoder)

        self.critic = Critic(self.state_dim, action_dim, hidden_dims, activation=activation, use_popart=use_popart, beta=beta)
        self.actor = Actor(self.state_dim, action_dim, hidden_dims, activation=activation)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32),
                                      requires_grad=True)

        self.modules = Container()
        self.modules.state_encoder = self.state_encoder
        self.modules.critic = self.critic
        self.modules.actor = self.actor
        self.modules.params = nn.ParameterDict({'log_alpha': self.log_alpha})
        self.modules.to(self.model_device)

        self.state_encoder.share_memory()
        self.actor.share_memory()
        self.collector = self.COLLECTOR(env_func=env_func,
                                        env_kwargs=env_kwargs,
                                        state_encoder=self.state_encoder,
                                        actor=self.actor,
                                        n_samplers=n_samplers,
                                        n_bootstrap_step=n_bootstrap_step,
                                        buffer_capacity=buffer_capacity,
                                        batch_size=batch_size,
                                        n_frames=n_frames,
                                        n_step_return=n_bootstrap_step,
                                        devices=self.sampler_devices,
                                        random_seed=random_seed)

    def print_info(self, file=None):
        print(f'state_dim = {self.state_dim}', file=file)
        print(f'action_dim = {self.action_dim}', file=file)
        print(f'device = {self.model_device}', file=file)
        print(f'buffer_capacity = {self.replay_buffer.capacity}', file=file)
        print(f'n_samplers = {self.collector.n_samplers}', file=file)
        print(f'sampler_devices = {list(map(str, self.collector.devices))}', file=file)
        print('Modules:', self.modules, file=file)

    @property
    def replay_buffer(self):
        return self.collector.replay_buffer

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_episode_video=False, log_dir=None):
        self.collector.sample(n_episodes, max_episode_steps, deterministic, random_sample,
                              render, log_episode_video, log_dir)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_episode_video=False, log_dir=None):
        samplers = self.collector.async_sample(n_episodes, max_episode_steps,
                                               deterministic, random_sample,
                                               render, log_episode_video, log_dir)
        return samplers

    def train(self, mode=True):
        if self.training != mode:
            self.training = mode
            self.modules.train(mode=mode)
        self.collector.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def save_model(self, path):
        self.modules.save_model(path)

    def load_model(self, path, strict=True):
        return self.modules.load_model(path, strict=strict)


class Trainer(ModelBase):
    def __init__(self, env_func, env_kwargs, state_encoder,
                 state_dim, action_dim, hidden_dims, activation, n_past_actions, n_bootstrap_step,
                 initial_alpha, critic_lr, actor_lr, alpha_lr, weight_decay, actor_update_frequency,
                 use_popart, beta, n_samplers, buffer_capacity, batch_size, n_frames, devices,
                 sampler_devices, separate_encoder, random_seed=0):
        super().__init__(env_func, env_kwargs, state_encoder,
                         state_dim, action_dim, hidden_dims, activation, n_past_actions, n_bootstrap_step,
                         initial_alpha, use_popart, beta, n_samplers, buffer_capacity, batch_size, n_frames,
                         devices, sampler_devices, separate_encoder, random_seed)

        self.dtype = torch.float32

        self.target_critic = clone_network(src_net=self.critic, device=self.model_device)
        self.target_critic.eval().requires_grad_(False)

        self.critic_criterion = nn.MSELoss()

        self.global_step = 0
        self.actor_update_frequency = actor_update_frequency
        self.use_popart = use_popart
        self.amp_dtype = torch.float32

        if isinstance(self.state_encoder.encoder, VAEBase):
            self.optimizer = optim.Adam(itertools.chain(self.critic.parameters(),
                                                        self.actor.parameters()),
                                        lr=critic_lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                        self.critic.parameters(),
                                                        self.actor.parameters()),
                                        lr=critic_lr, weight_decay=weight_decay)

        init_optimizer(self.optimizer)
        self.actor_loss_weight = actor_lr / critic_lr

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train(mode=True)

    def update_sac(self, actor_state, critic_state, action, reward, 
                   actor_next_state, critic_next_state, done,
                   normalize_rewards=True, reward_scale=1.0,
                   adaptive_entropy=True, target_entropy=-2.0,
                   clip_gradient=False, gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        # Normalize rewards
        if normalize_rewards:
            with torch.no_grad():
                reward = reward_scale * (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob, _ = self.actor.evaluate(actor_state)
        if adaptive_entropy:
            # equation 18
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()

        if adaptive_entropy:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Train Q function
        with torch.no_grad():
            new_next_action, next_log_prob, _ = self.actor.evaluate(actor_next_state)

            # use min to mitigate bias
            target_q_min = torch.min(*self.target_critic(critic_next_state, new_next_action))
            # target_q_min is V(s_t+1) through equation 3
            target_q_min -= alpha * next_log_prob
            # equation 5
            target_q_value = reward + (1 - done) * gamma * target_q_min

        # variables for non-normalize reward and support pop art reward
        normalized_target = False
        target_q_value1 = target_q_value
        target_q_value2 = target_q_value
        if self.use_popart:
            self.critic.update_popart_parameters(target_q_value)
            normalized_target = True
            target_q_value1, target_q_value2 = self.critic.get_normalized_targets(target_q_value)

        predicted_q_value_1, predicted_q_value_2 = self.critic(critic_state, action, normalize=normalized_target)

        critic_loss_1 = self.critic_criterion(predicted_q_value_1, target_q_value1)
        critic_loss_2 = self.critic_criterion(predicted_q_value_2, target_q_value2)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2.0

        if self._should_update_actor():
            # Train policy function
            predicted_new_q_value = torch.min(*self.critic(critic_state, new_action))
            predicted_new_q_value_critic_grad_only = torch.min(*self.critic(critic_state, new_action.detach()))
            actor_loss = (alpha * log_prob - predicted_new_q_value).mean()
            actor_loss_unbiased = actor_loss + predicted_new_q_value_critic_grad_only.mean()

        if self._should_update_actor():
            loss = critic_loss + self.actor_loss_weight * actor_loss_unbiased
        else:
            loss = critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        if clip_gradient:
            clip_grad_norm(self.optimizer)
        self.optimizer.step()

        if self._should_update_actor():
            # Soft update the target value net
            sync_params(src_net=self.critic, dst_net=self.target_critic, soft_tau=soft_tau)

        info = {
            'critic_loss': critic_loss.item(),
            'temperature_parameter': alpha.item()
        }
        if self._should_update_actor():
            info['actor_loss'] = actor_loss.item()

        self.global_step += 1

        return info

    def update(self, batch_size, normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
               clip_gradient=False, gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        self.train()

        # size: (batch_size, item_size)
        actor_state, critic_state, action, reward, actor_next_state, critic_next_state, done = self.prepare_batch(batch_size)

        return self.update_sac(actor_state, critic_state, action, reward, actor_next_state, critic_next_state, done,
                               normalize_rewards, reward_scale,
                               adaptive_entropy, target_entropy,
                               clip_gradient, gamma, soft_tau, epsilon)

    def prepare_batch(self, batch_size):
        if self.n_past_actions > 1:
            observation, additional_state, action, reward, next_observation, next_additional_state, done \
                = self.replay_buffer.sample(batch_size, normalize=True, device=self.model_device)
        else:
            observation, action, reward, next_observation, done = self.replay_buffer.sample(batch_size)

            observation = torch.tensor(observation, dtype=torch.float32, device=self.model_device)
            next_observation = torch.tensor(next_observation, dtype=torch.float32, device=self.model_device)

            action = torch.tensor(action, dtype=torch.float32, device=self.model_device)

            reward = torch.tensor(reward, dtype=torch.float32, device=self.model_device)
            done = torch.tensor(done, dtype=torch.float32, device=self.model_device)

        if isinstance(self.state_encoder.encoder, VAEBase):
            with torch.no_grad():
                self.state_encoder.eval()
                # observation size is [32, 256, 512, 3]
                state = encode_vae_observation(observation,
                                            self.state_encoder,
                                            device=self.model_device,
                                            output_device=self.model_device,
                                            normalize=False)
                next_state = encode_vae_observation(next_observation,
                                                    self.state_encoder,
                                                    device=self.model_device,
                                                    output_device=self.model_device,
                                                    normalize=False)
        else:
            if self.separate_encoder:
                actor_state, critic_state = self.state_encoder(observation)
                with torch.no_grad():
                    actor_next_state, critic_next_state = self.state_encoder(next_observation)
            else:
                state = self.state_encoder(observation, additional_state.view(batch_size, -1))
                with torch.no_grad():
                    next_state = self.state_encoder(next_observation, next_additional_state.view(batch_size, -1))

        if self.n_past_actions > 1:
            if self.separate_encoder:
                actor_state = torch.cat((actor_state, additional_state.view(batch_size, -1)), dim=1)
                critic_state = torch.cat((critic_state, additional_state.view(batch_size, -1)), dim=1)
                actor_next_state = torch.cat((actor_next_state, next_additional_state.view(batch_size, -1)), dim=1)
                critic_next_state = torch.cat((critic_next_state, next_additional_state.view(batch_size, -1)), dim=1)
            else:
                pass
        if not self.separate_encoder:
            actor_state = critic_state = state
            actor_next_state = critic_next_state = next_state

        # size: (batch_size, item_size)
        return actor_state, critic_state, action, reward, actor_next_state, critic_next_state, done

    def _should_update_actor(self):
        return self.global_step % self.actor_update_frequency == 0

    def save_model(self, path):
        optimizer = None
        if hasattr(self, 'optimizer'):
            optimizer = self.optimizer

        self.modules.save_model(path, optimizer=optimizer)

    def load_model(self, path, strict=True, load_weight_only=False):
        load_result = super().load_model(path=path, strict=strict)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval().requires_grad_(False)

        state_dict = torch.load(path, map_location=self.model_device)
        if 'optimizer' in state_dict and not load_weight_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if 'alpha_optimizer' in state_dict and not load_weight_only:
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])

        return load_result


class Tester(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval()
        self.modules.requires_grad_(False)


class RenderTester(object):
    ''' For single render test '''
    STATE_ENCODER_WRAPPER = StateEncoderWrapper

    def __init__(self, env_func, env_kwargs, state_encoder,
                 state_dim, action_dim, hidden_dims, activation, n_past_actions,
                 initial_alpha, n_samplers, buffer_capacity,
                 devices, sampler_devices, random_seed=0):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)
        self.sampler_devices = sampler_devices

        self.n_past_actions = n_past_actions
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = self.STATE_ENCODER_WRAPPER(state_encoder)

        self.critic = Critic(self.state_dim, action_dim, hidden_dims, activation=activation)
        self.actor = Actor(self.state_dim, action_dim, hidden_dims, activation=activation)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32),
                                      requires_grad=True)

        self.modules = Container()
        self.modules.state_encoder = self.state_encoder
        self.modules.critic = self.critic
        self.modules.actor = self.actor
        self.modules.params = nn.ParameterDict({'log_alpha': self.log_alpha})
        self.modules.to(self.model_device)

        self.env = env_func(**env_kwargs)
        self.env.seed(random_seed)
        self.eval()

    def print_info(self, file=None):
        print(f'state_dim = {self.state_dim}', file=file)
        print(f'action_dim = {self.action_dim}', file=file)
        print(f'device = {self.model_device}', file=file)
        print('Modules:', self.modules, file=file)

    def train(self, mode=True):
        if self.training != mode:
            self.training = mode
            self.modules.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def save_model(self, path):
        self.modules.save_model(path)

    def load_model(self, path, strict=True):
        return self.modules.load_model(path, strict=strict)
