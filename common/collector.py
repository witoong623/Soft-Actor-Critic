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
import torch_tensorrt
import tqdm
from PIL import Image, ImageDraw, ImageFont
from setproctitle import setproctitle
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer, EpisodeReplayBuffer, EfficientReplayBuffer, PrioritizedReplayBuffer
from common.utils import clone_network, sync_params, normalize_image, \
    normalize_grayscale_image, ObservationStacker, label_mask_to_color_mask
from common.carla_environment.action_sampler import CarlaBiasActionSampler, CarlaPIDLongitudinalSampler, CarlaPerfectActionSampler
from extra.user_episode_loader import UserEpisodeAdder


__all__ = ['Collector', 'EpisodeCollector']


MEAN = np.tile([0.3171, 0.3183, 0.3779], 2)
STD = np.tile([0.1406, 0.0594, 0.0925], 2)


RGB_MEAN = np.array([0.4203, 0.4078, 0.3611])
RGB_STD = np.array([0.2669, 0.2638, 0.2579])


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, sampler_lock,
                 running_event, event, next_sampler_event,
                 env_func, env_kwargs, state_encoder, actor,
                 eval_only, replay_buffer, n_frames,
                 n_total_steps, episode_steps, episode_rewards,
                 n_bootstrap_step, n_episodes, max_episode_steps,
                 deterministic, random_sample, render, log_episode_video,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}', daemon=True)

        self.rank = rank
        self.sampler_lock = sampler_lock
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

        self.n_frames = n_frames
        self.replay_buffer = replay_buffer
        self.n_total_steps = n_total_steps
        self.episode_steps = episode_steps
        self.episode_rewards = episode_rewards
        self.n_bootstrap_step = n_bootstrap_step

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
        self.log_episode_video_frequency = 50

        self.log_dir = log_dir

        self.episode = 0
        self.trajectory = []
        self.frames = []
        self.image_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", 16)
        self.render()

    def run(self):
        setproctitle(title=self.name)

        self.seg_model = torch.jit.load('/home/witoon/thesis/code/thesis-code/segmentation-model/checkpoints/bisenetv2/best.ts', map_location=self.device)

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
            if self.random_sample:
                self.user_episode_adder = UserEpisodeAdder(self.n_episodes)

            while self.episode < self.n_episodes:
                self.episode += 1

                if not (self.eval_only or self.random_sample):
                    sync_params(src_net=self.shared_state_encoder, dst_net=self.state_encoder)
                    sync_params(src_net=self.shared_actor, dst_net=self.actor)

                obs_stacker = ObservationStacker(n_frames=self.n_frames, stack_axis=2)

                episode_reward = 0
                episode_steps = 0
                if self.state_encoder is not None:
                    self.state_encoder.reset()
                observation = self.env.reset()

                extra_state = None
                if hasattr(self.env, 'first_extra_state'):
                    extra_state = self.env.first_extra_state

                if self.random_sample:
                    if self.env.is_AIT_map():
                        action_sampler = self._create_ait_action_sampler()
                    else:
                        action_sampler = self._create_town7_action_sampler()

                self.render()
                self.frames.clear()
                self.save_frame(step=0, reward=np.nan, episode_reward=0.0)
                for step in range(self.max_episode_steps):
                    if self.random_sample:
                        action, done_sample = action_sampler.sample()
                    else:
                        # observation shape (H, W, C)
                        stacked_obs = obs_stacker.get_new_observation(observation)
                        normalized_obs = normalize_image(stacked_obs, MEAN, STD).transpose((2, 0, 1))
                        extra_state_tensor = torch.tensor(extra_state, dtype=amp_dtype, device=self.device)
                        state = self.state_encoder.encode(normalized_obs, extra_state_tensor, return_tensor=True, data_dtype=amp_dtype)

                        action = self.actor.get_action(state, deterministic=self.deterministic)

                    next_observation, reward, done, info = self.env.step(action)

                    next_extra_state = info.get('extra_state', None)
                    should_stop = info.get('should_stop', False)

                    episode_reward += reward
                    episode_steps += 1

                    self.save_frame(step=episode_steps, reward=reward, episode_reward=episode_reward)
                    self.add_transaction(observation, extra_state, action, reward, done)

                    observation = next_observation
                    extra_state = next_extra_state

                    if done or (self.random_sample and done_sample) or should_stop:
                        break

                    if step > self.max_episode_steps * 0.2:
                        # wait for signal from Collector to stop or continue
                        self.running_event.wait()
                        with self.sampler_lock:
                            self.save_trajectory(is_end=False)

                # wait for signal from Collector to stop or continue
                self.running_event.wait()
                # wait for previous Sampler to set
                self.event.wait(timeout=self.timeout)
                with self.sampler_lock:
                    self.save_trajectory(is_end=True)
                    self.save_stat(episode_steps, episode_reward)
                self.event.clear()

                # set next Sampler to continue save trajectory
                self.next_sampler_event.set()

                self.save_sampler_log(episode_steps, episode_reward)

                if self.random_sample and \
                    self.env.is_AIT_map() and \
                    self.user_episode_adder.should_add_user_episode(self.episode):
                    self._add_user_episode()
        except KeyboardInterrupt:
            self.close()
            return

        self.env.close()

        if self.writer is not None:
            self.writer.close()

        self.seg_model = None

    def add_transaction(self, observation, extra_state, action, reward, done):
        self.trajectory.append((observation, extra_state, action, reward, done))

    def save_trajectory(self, is_end=True):
        self._transform_observations()
        self.replay_buffer.extend(self.trajectory, is_end)
        self.trajectory.clear()

    def save_stat(self, episode_steps, episode_reward):
        self.n_total_steps.value += episode_steps
        self.episode_steps.append(episode_steps)
        self.episode_rewards.append(episode_reward)

    def save_sampler_log(self, episode_steps, episode_reward):
        if self.writer is None:
            return

        average_reward = episode_reward / episode_steps
        self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=self.episode)
        self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=self.episode)
        self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=self.episode)
        self.writer.add_scalar(tag='sample/milestone', scalar_value=self.env.get_latest_milestone(), global_step=self.episode)
        self.log_video()
        self.writer.flush()

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
        if not self.random_sample and self.log_episode_video and self.episode % self.log_episode_video_frequency == 0:
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
        if self.writer is not None and self.log_episode_video and self.episode % self.log_episode_video_frequency == 0:
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

    def _create_ait_action_sampler(self):
        action_sampler = CarlaBiasActionSampler(forward_only=False, use_brake=True, try_correction=True)
        return action_sampler

    def _create_town7_action_sampler(self):
        if not hasattr(self, 'does_perfect_sample'):
            self.does_perfect_sample = False

        if random.random() > 0.95 or \
            (self.n_episodes - self.episode == 1 and not self.does_perfect_sample):
            self.does_perfect_sample = True
            action_sampler = CarlaPerfectActionSampler(self.env)
        else:
            action_sampler = CarlaBiasActionSampler(forward_only=False, use_brake=True, try_correction=True)

        return action_sampler

    def _add_user_episode(self):
        user_episode, episode_steps, episode_reward = self.user_episode_adder.get_episode(self.episode)
        self.trajectory.extend(user_episode)
        self.save_trajectory(is_end=True)

        self.save_stat(episode_steps, episode_reward)
        self.save_sampler_log(episode_steps, episode_reward)

    def _add_all_user_episode(self):
        for chunk, chunk_steps, chunk_reward in self.user_episode_adder.get_all_episodes():
            self.replay_buffer.extend(chunk, is_end=True)

            self.save_stat(chunk_steps, chunk_reward)
            self.save_sampler_log(chunk_steps, chunk_reward)

    def _transform_observations(self):
        batch_size = 8
        for b in range(0, len(self.trajectory), batch_size):
            upper = min(len(self.trajectory) - b, batch_size)
            np_img_batch = np.stack([normalize_image(self.trajectory[b+i][0], RGB_MEAN, RGB_STD).transpose((2, 0, 1)) for i in range(upper)])
            batch_tensor = torch.tensor(np_img_batch, dtype=torch.float16, device=self.device)
            out_tensor = self.seg_model(batch_tensor)
            out_np = out_tensor.cpu().numpy()

            for i in range(upper):
                label_mask = np.argmax(out_np[i], axis=0)
                self.trajectory[b+i] = (label_mask_to_color_mask(label_mask),) + self.trajectory[b+i][1:]


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
                 batch_size, n_frames, n_step_return, prioritize_replay,
                 devices, random_seed):
        self.manager = mp.Manager()
        self.running_event = self.manager.Event()
        self.running_event.set()
        self.total_steps = self.manager.Value('L', 0)
        self.episode_steps = self.manager.list()
        self.episode_rewards = self.manager.list()
        self.sampler_lock = self.manager.Lock()

        self.state_encoder = state_encoder
        self.actor = actor
        self.eval_only = False

        self.n_frames = n_frames
        self.n_bootstrap_step = n_bootstrap_step
        self.n_samplers = n_samplers

        if prioritize_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, batch_size=batch_size,
                                                         n_frames=n_frames, n_step_return=n_step_return,
                                                         frame_stack_mode='concatenate', frame_stack_axis=2,
                                                         list_initializer=self.manager.list, dict_initializer=self.manager.dict,
                                                         Value=self.manager.Value, Lock=self.manager.Lock)
        else:
            self.replay_buffer = EfficientReplayBuffer(capacity=buffer_capacity, batch_size=batch_size,
                                                       n_frames=n_frames, n_step_return=n_step_return,
                                                       frame_stack_mode='concatenate', frame_stack_axis=2,
                                                       list_initializer=self.manager.list, dict_initializer=self.manager.dict,
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
            sampler = self.SAMPLER(rank, self.n_samplers, self.sampler_lock,
                                   self.running_event, events[rank], events[(rank + 1) % self.n_samplers],
                                   self.env_func, self.env_kwargs, self.state_encoder, self.actor,
                                   self.eval_only, self.replay_buffer, self.n_frames,
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
