import itertools
import math
import os
import random
import time
from functools import lru_cache

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.cuda.amp as amp
import tqdm
from PIL import Image, ImageDraw, ImageFont
from setproctitle import setproctitle
from torch.utils.tensorboard import SummaryWriter

from .buffer import ReplayBuffer, EpisodeReplayBuffer
from .utils import clone_network, sync_params, CarlaBiasActionSampler
from .carla_environment.environment import CarlaPerfectActionSampler


__all__ = ['Collector', 'EpisodeCollector']


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, lock,
                 running_event, event, next_sampler_event,
                 env_func, env_kwargs, state_encoder, actor,
                 eval_only, replay_buffer,
                 n_total_steps, episode_steps, episode_rewards,
                 n_bootstrap_step, n_episodes, max_episode_steps,
                 deterministic, random_sample, render, log_episode_video,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}', daemon=True)

        self.rank = rank
        self.lock = lock
        self.running_event = running_event
        self.event = event
        self.next_sampler_event = next_sampler_event
        self.timeout = 60.0 * n_samplers

        self.env = None
        self.env_func = env_func
        self.env_kwargs = env_kwargs
        self.random_seed = random_seed

        self.shared_state_encoder = state_encoder
        self.shared_actor = actor
        self.state_encoder = None
        self.actor = None
        self.device = device
        self.eval_only = eval_only

        self.replay_buffer = replay_buffer
        self.n_total_steps = n_total_steps
        self.episode_steps = episode_steps
        self.episode_rewards = episode_rewards
        self.n_bootstrap_step = n_bootstrap_step
        self._cumulative_discount_vector = np.array([math.pow(0.99, n) for n in range(self.n_bootstrap_step)])

        if np.isinf(n_episodes):
            self.n_episodes = np.inf
        else:
            self.n_episodes = n_episodes // n_samplers
            if rank < n_episodes % n_samplers:
                self.n_episodes += 1

        # TODO: delete this condition if I don't use lap env
        if random_sample:
            # fix random sample to 1000 steps
            self.max_episode_steps = 1000
        else:
            self.max_episode_steps = max_episode_steps

        self.deterministic = deterministic
        self.random_sample = random_sample
        self.render_env = (render and rank == 0)
        self.log_episode_video = (log_episode_video and rank == 0)

        self.log_dir = log_dir

        self.episode = 0
        self.does_perfect_sample = False
        self.trajectory = []
        self.frames = []
        self.image_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", 6)
        self.render()

    def run(self):
        setproctitle(title=self.name)

        self.env = self.env_func(**self.env_kwargs)
        self.env.seed(self.random_seed)
        amp_dtype = torch.float32

        if not self.random_sample:
            self.state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
            self.actor = clone_network(src_net=self.shared_actor, device=self.device)
            self.state_encoder.eval().requires_grad_(False)
            self.actor.eval().requires_grad_(False)

        self.episode = 0
        try:
            while self.episode < self.n_episodes:
                self.episode += 1

                if not (self.eval_only or self.random_sample):
                    sync_params(src_net=self.shared_state_encoder, dst_net=self.state_encoder)
                    sync_params(src_net=self.shared_actor, dst_net=self.actor)

                episode_reward = 0
                episode_steps = 0
                self.trajectory.clear()
                if self.state_encoder is not None:
                    self.state_encoder.reset()
                observation = self.env.reset()

                additional_state = None
                if hasattr(self.env, 'first_additional_state'):
                    additional_state = self.env.first_additional_state

                if self.random_sample:
                    if random.random() > 0.97 or \
                        (self.n_episodes - self.episode == 1 and not self.does_perfect_sample):
                        self.does_perfect_sample = True
                        action_sampler = CarlaPerfectActionSampler(self.env)
                    else:
                        action_sampler = CarlaBiasActionSampler()

                self.render()
                self.frames.clear()
                self.save_frame(step=0, reward=np.nan, episode_reward=0.0)
                for step in range(self.max_episode_steps):
                    if self.random_sample:
                        action, done_sample = action_sampler.sample()
                    else:
                        with amp.autocast(dtype=amp_dtype, enabled=False):
                            # observation shape (H, W, C)
                            state = self.state_encoder.encode(observation, return_tensor=True)

                        if additional_state is not None:
                            # use amp_dtype for additional state to match state dtype
                            additional_state_tensor = torch.tensor(additional_state, dtype=amp_dtype, device=self.device)
                            state = torch.cat((state, additional_state_tensor))

                        with amp.autocast(dtype=amp_dtype, enabled=False):
                            action = self.actor.get_action(state, deterministic=self.deterministic)

                    next_observation, reward, done, info = self.env.step(action)
                    if 'additional_state' in info:
                        next_additional_state = info['additional_state']

                    episode_reward += reward
                    episode_steps += 1

                    self.save_frame(step=episode_steps, reward=reward, episode_reward=episode_reward)
                    self.add_transaction(observation, action, reward, next_observation, done, additional_state, next_additional_state)

                    observation = next_observation
                    additional_state = next_additional_state

                    if done or (self.random_sample and done_sample):
                        break

                # wait for signal from Collector to stop or continue
                self.running_event.wait()
                # wait for previous Sampler to set
                self.event.wait(timeout=self.timeout)
                with self.lock:
                    self.save_trajectory()
                    self.n_total_steps.value += episode_steps
                    self.episode_steps.append(episode_steps)
                    self.episode_rewards.append(episode_reward)
                self.event.clear()
                # set next Sampler to continue save trajectory
                self.next_sampler_event.set()
                if self.writer is not None:
                    average_reward = episode_reward / episode_steps
                    self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=self.episode)
                    self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=self.episode)
                    self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=self.episode)
                    self.writer.add_scalar(tag='sample/milestone', scalar_value=self.env.get_latest_milestone(), global_step=self.episode)
                    self.log_video()
                    self.writer.flush()
        except KeyboardInterrupt:
            self.close()
            return

        self.env.close()

        if self.writer is not None:
            self.writer.close()

    def add_transaction(self, observation, action, reward, next_observation, done, additional_state, next_additional_state):
        if self.n_bootstrap_step > 1:
            # for easy calculation in bootstrapping
            wrapped_reward = reward
            wrapped_done = done
        else:
            wrapped_reward = [reward]
            wrapped_done = [done]

        self.trajectory.append((observation, additional_state, action, wrapped_reward, next_observation, next_additional_state, wrapped_done))

    def save_trajectory(self):
        if self.n_bootstrap_step > 1:
            self._save_bootstrapped_trajectory()
        else:
            self.replay_buffer.extend(self.trajectory)

    def _save_bootstrapped_trajectory(self):
        reward_array = np.array([t[3] for t in self.trajectory])
        done_array = np.array([t[-1] for t in self.trajectory])
        new_trajectory = []

        for step_idx in range(len(self.trajectory)):
            trajectory_indexes = [(step_idx + i) % len(self.trajectory) for i in range(self.n_bootstrap_step)]

            trajectory_dones = done_array[trajectory_indexes]
            is_done = trajectory_dones.any()

            if is_done:
                # because argmax return index not count, +1 make it count
                trajectory_len = np.argmax(trajectory_dones.astype(bool), axis=0) + 1
            else:
                # prevent len is more than remaining trajectory, -1 to convert to base 0 index
                trajectory_len = min(self.n_bootstrap_step, len(self.trajectory) - step_idx - 1)

            next_step_idx = step_idx + trajectory_len

            observation = self.trajectory[step_idx][0]
            additional_state = self.trajectory[step_idx][1]
            action = self.trajectory[step_idx][2]

            try:
                reward = np.dot(reward_array[step_idx:next_step_idx], self._cumulative_discount_vector)
            except ValueError:
                discount_vector = np.array([math.pow(0.99, n) for n in range(trajectory_len)])
                reward = np.dot(reward_array[step_idx:next_step_idx], discount_vector)

            try:
                next_observation = self.trajectory[next_step_idx][4]
                next_additional_state = self.trajectory[next_step_idx][5]
            except IndexError:
                if is_done:
                    next_step_idx -= 1
                    # choose last observation, this won't be used anyway if this is terminal state
                    next_observation = self.trajectory[next_step_idx][4]
                    next_additional_state = self.trajectory[next_step_idx][5]
                else:
                    raise

            done = is_done

            new_trajectory.append((observation, additional_state, action, [float(reward)], next_observation, next_additional_state, [done]))

        self.replay_buffer.extend(new_trajectory)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass

    def render(self, mode='human', **kwargs):
        if self.render_env:
            try:
                return self.env.render(mode=mode, **kwargs)
            except Exception:
                pass

    def save_frame(self, step, reward, episode_reward):
        if not self.random_sample and self.log_episode_video and self.episode % 50 == 0:
            try:
                img = self.env.render(mode='rgb_array')
            except Exception:
                pass
            else:
                text = (f'step           = {step}\n'
                        f'reward         = {reward:+.3f}\n'
                        f'episode reward = {episode_reward:+.3f}')
                img = Image.fromarray(img, mode='RGB')
                draw = ImageDraw.Draw(img)
                draw.multiline_text(xy=(3, 3), text=text, font=self.image_font, fill=(255, 0, 0))
                img = np.asanyarray(img, dtype=np.uint8)
                self.frames.append(img)

    def log_video(self):
        if self.writer is not None and self.log_episode_video and self.episode % 50 == 0:
            try:
                video = np.stack(self.frames).transpose((0, 3, 1, 2))
                video = np.expand_dims(video, axis=0)
                self.writer.add_video(tag='sample/episode', vid_tensor=video, global_step=self.episode, fps=10)
            except ValueError:
                pass

    @property
    @lru_cache(maxsize=None)
    def writer(self):
        if not self.random_sample and self.log_dir is not None:
            return SummaryWriter(log_dir=os.path.join(self.log_dir, self.name), comment=self.name)
        else:
            return None


class EpisodeSampler(Sampler):
    def add_transaction(self, observation, action, reward, next_observation, done):
        self.trajectory.append((observation, action, [reward], [done]))

    def save_trajectory(self):
        # each component is stacked from t_0 to t_T
        self.replay_buffer.push(*tuple(map(np.stack, zip(*self.trajectory))))


class Collector(object):
    SAMPLER = Sampler
    REPLAY_BUFFER = ReplayBuffer

    def __init__(self, env_func, env_kwargs, state_encoder, actor,
                 n_samplers, n_bootstrap_step, buffer_capacity,
                 devices, random_seed):
        self.manager = mp.Manager()
        self.running_event = self.manager.Event()
        self.running_event.set()
        self.total_steps = self.manager.Value('L', 0)
        self.episode_steps = self.manager.list()
        self.episode_rewards = self.manager.list()
        self.lock = self.manager.Lock()

        self.state_encoder = state_encoder
        self.actor = actor
        self.eval_only = False

        self.n_bootstrap_step = n_bootstrap_step
        self.n_samplers = n_samplers
        self.replay_buffer = self.REPLAY_BUFFER(capacity=buffer_capacity, initializer=self.manager.list,
                                                Value=self.manager.Value, Lock=self.manager.Lock)

        self.env_func = env_func
        self.env_kwargs = env_kwargs
        self.devices = [device for _, device in zip(range(n_samplers), itertools.cycle(devices))]
        self.random_seed = random_seed

        self.samplers = []

    @property
    def n_episodes(self):
        return len(self.episode_steps)

    @property
    def n_total_steps(self):
        return self.total_steps.value

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_episode_video=False, log_dir=None):
        self.resume()

        events = [self.manager.Event() for i in range(self.n_samplers)]
        for event in events:
            event.clear()
        events[0].set()

        for rank in range(self.n_samplers):
            sampler = self.SAMPLER(rank, self.n_samplers, self.lock,
                                   self.running_event, events[rank], events[(rank + 1) % self.n_samplers],
                                   self.env_func, self.env_kwargs, self.state_encoder, self.actor,
                                   self.eval_only, self.replay_buffer,
                                   self.total_steps, self.episode_steps, self.episode_rewards,
                                   self.n_bootstrap_step, n_episodes, max_episode_steps,
                                   deterministic, random_sample, render, log_episode_video,
                                   self.devices[rank], self.random_seed + rank, log_dir)
            sampler.start()
            self.samplers.append(sampler)

        return self.samplers

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_episode_video=False, log_dir=None):
        n_initial_episodes = self.n_episodes

        self.async_sample(n_episodes, max_episode_steps, deterministic, random_sample,
                          render, log_episode_video, log_dir)

        pbar = tqdm.tqdm(total=n_episodes, desc='Sampling')
        while True:
            n_new_episodes = self.n_episodes - n_initial_episodes
            if n_new_episodes > pbar.n:
                pbar.n = n_new_episodes
                pbar.set_postfix({'buffer_size': self.replay_buffer.size})
                if pbar.n >= n_episodes:
                    break
            else:
                time.sleep(0.1)

        self.join()

    def join(self):
        for sampler in self.samplers:
            sampler.join()
            sampler.close()
        self.samplers.clear()

    def terminate(self):
        self.pause()
        for sampler in self.samplers:
            if sampler.is_alive():
                try:
                    sampler.terminate()
                except Exception:
                    pass
        self.join()

    def pause(self):
        self.running_event.clear()

    def resume(self):
        self.running_event.set()

    def train(self, mode=True):
        self.eval_only = (not mode)
        return self

    def eval(self):
        return self.train(mode=False)

    @property
    def are_samplers_running(self):
        ''' return True if at least one sampler running, otherwise False '''
        return any([sampler.is_alive() for sampler in self.samplers])


class EpisodeCollector(Collector):
    SAMPLER = EpisodeSampler
    REPLAY_BUFFER = EpisodeReplayBuffer
