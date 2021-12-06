from collections import deque

import gym
import numpy as np
import torchvision.transforms as transforms
from gym.spaces import Box
from gym.wrappers import TimeLimit


__all__ = [
    'build_env', 'initialize_environment',
    'FlattenedAction', 'NormalizedAction',
    'FlattenedObservation', 'VisionObservation', 'ConcatenatedObservation'
]

try:
    import pybullet_envs
except ImportError:
    pass

try:
    import mujoco_py
except Exception:
    pass


def build_env(**kwargs):
    env = gym.make(kwargs['name'])
    env.seed(kwargs['random_seed'])

    env = NormalizedAction(FlattenedAction(env))
    if kwargs['vision_observation'] and kwargs['name'] == 'CarRacing-v0':
        env = CarRacingEnvWrapper(env, n_frames=kwargs['n_frames'], n_repeat_actions=kwargs['n_repeat_actions'],
                                  n_past_actions=kwargs['n_past_actions'], to_grayscale=kwargs['encoder_arch'] != 'VAE')
    elif kwargs['vision_observation']:
        env = VisionObservation(env, image_size=(kwargs['image_size'], kwargs['image_size']))
    else:
        env = FlattenedObservation(env)

    if kwargs['n_frames'] > 1 and kwargs['name'] != 'CarRacing-v0' :
        env = ConcatenatedObservation(env, n_frames=kwargs['n_frames'], dim=0)

    max_episode_steps = kwargs['max_episode_steps']
    try:
        max_episode_steps = min(max_episode_steps, env.spec.max_episode_steps)
    except AttributeError:
        pass
    except TypeError:
        pass
    # env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def initialize_environment(config):
    config.env_func = build_env
    config.env_kwargs = config.build_from_keys(['vision_observation',
                                                'image_size',
                                                'n_frames',
                                                'max_episode_steps',
                                                'random_seed',
                                                'n_repeat_actions',
                                                'n_past_actions',
                                                'encoder_arch'])
    config.env_kwargs.update(name=config.env)

    with config.env_func(**config.env_kwargs) as env:
        print(f'env = {env}')
        print(f'observation_space.shape = {env.observation_space.shape}')
        print(f'action_space.shape = {env.action_space.shape}')

        config.observation_dim = env.observation_space.shape[0]
        config.action_dim = env.action_space.shape[0]
        try:
            config.max_episode_steps = min(config.max_episode_steps, env.spec.max_episode_steps)
        except AttributeError:
            pass
        except TypeError:
            pass
        config.env_kwargs['max_episode_steps'] = config.max_episode_steps
        if config.RNN_encoder:
            assert config.step_size <= config.max_episode_steps


class FlattenedAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.action_space = Box(low=self.env.action_space.low.ravel(),
                                high=self.env.action_space.high.ravel(),
                                dtype=np.float32)

    def action(self, action):
        return np.ravel(action)

    def reverse_action(self, action):
        return np.reshape(action, self.env.action_space.shape)


class NormalizedAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.action_space = Box(low=-1.0,
                                high=1.0,
                                shape=self.env.action_space.shape,
                                dtype=np.float32)

    def action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high

        action = low + 0.5 * (action + 1.0) * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high

        action = np.clip(action, low, high)
        action = 2.0 * (action - low) / (high - low) - 1.0

        return action


class FlattenedObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.observation_space = Box(low=self.env.observation_space.low.ravel(),
                                     high=self.env.observation_space.high.ravel(),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.ravel(observation)


class VisionObservation(gym.ObservationWrapper):
    def __init__(self, env, image_size=(128, 128)):
        super().__init__(env=env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(3, *image_size), dtype=np.float32)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=image_size),
            transforms.ToTensor()
        ])

        self.unwrapped_observation_space = self.env.observation_space
        self.unwrapped_observation = None

    def observation(self, observation):
        self.unwrapped_observation = observation

        obs = self.render(mode='rgb_array')
        obs = self.transform(obs).cpu().detach().numpy()

        return obs


class NaturalVisionObservation(gym.ObservationWrapper):
    def __init__(self, env, image_size=(128, 128)):
        super().__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(3, *image_size), dtype=np.float32)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=image_size),
            transforms.Lambda(lambd=lambda img: np.array(img).transpose(2, 0, 1) / 255)
        ])

        print(f'NaturalVisionObservation observation_space: {self.env.observation_space.shape}')

    def observation(self, observation):
        return self.transform(observation)


class ConcatenatedObservation(gym.ObservationWrapper):
    def __init__(self, env, n_frames=3, dim=0):
        super().__init__(env=env)

        self.observation_space = Box(low=np.concatenate([self.env.observation_space.low] * n_frames, axis=dim),
                                     high=np.concatenate([self.env.observation_space.high] * n_frames, axis=dim),
                                     dtype=self.env.observation_space.dtype)

        self.queue = deque(maxlen=n_frames)
        self.dim = dim
        print(f'ConcatenatedObservation observation_space: {self.observation_space.shape}')

    def reset(self, **kwargs):
        self.queue.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        while len(self.queue) < self.n_frames - 1:
            self.queue.append(observation)
        self.queue.append(observation)
        return np.concatenate(self.queue, axis=self.dim)

    @property
    def n_frames(self):
        return self.queue.maxlen


class RepeatActionEnvironment(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)

        self.repeat = repeat

    def reset(self):
        obs = super().reset()

        return obs

    def step(self, action):
        total_reward = 0
        infos = []

        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)

            total_reward += reward
            infos.append(info)
            if done:
                break

        return obs, total_reward, done, infos


class CarRacingEnvWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=4, n_repeat_actions=1, n_past_actions=1,
                 image_size=(96, 96), to_grayscale=True):
        super().__init__(env)
        # for single image, necessary as base for next calculation
        self.observation_space = Box(low=0.0, high=1.0, shape=(1, *image_size), dtype=np.float32)
        # for stack frame
        self.observation_space = Box(low=np.concatenate([self.observation_space.low] * n_frames, axis=0),
                                     high=np.concatenate([self.observation_space.high] * n_frames, axis=0),
                                     dtype=self.env.observation_space.dtype)

        self.n_frames = n_frames
        self.n_repeat_actions = n_repeat_actions
        self.n_past_actions = n_past_actions
        self.frames_queue = deque(maxlen=self.n_frames)

        if self.n_past_actions > 1:
            self.actions_queue = deque(maxlen=n_past_actions)

        if self.n_repeat_actions > 1:
            self._do_step = self._repeat_step
        else:
            self._do_step = self._step_frames_stacking

        if to_grayscale:
            def transform_gray_normalize(rgb_img):
                gray = np.dot(rgb_img[..., :], [0.299, 0.587, 0.114])
                return gray / 128. - 1.

            self.transform = transforms.Compose([
                transforms.Lambda(lambd=transform_gray_normalize)
            ])
            self.create_observation = lambda q: np.array(q)
        else:
            # identity
            self.transform = lambda x: x
            # self.create_observation = lambda q: np.concatenate(q, axis=2)
            self.create_observation = lambda q: np.array(q)

        self.render_inside = False

    def reset(self, **kwargs):
        self.avg_reward = self.reward_memory()
        self.frames_queue.clear()

        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames_queue.append(self.transform(obs))

        if self.n_past_actions > 1:
            self.actions_queue.clear()
            for _ in range(self.n_past_actions):
                self.actions_queue.append(np.array([0, 0, 0]))

        return self.create_observation(self.frames_queue)

    def step(self, action):
        obs, reward, done, info = self._do_step(action)

        if self.n_past_actions > 1:
            info['past_actions'] = np.array(self.actions_queue)
            self.actions_queue.append(action)
            info['next_past_actions'] = np.array(self.actions_queue)

        return obs, reward, done, info

    def _step_frames_stacking(self, action):
        if self.render_inside:
            self.env.render()

        obs, reward, die, info = self.env.step(action)
        # green penalty
        if np.mean(obs[:, :, 1]) > 185.0:
            reward -= 0.05
        # if no reward recently, end the episode
        done = True if self.avg_reward(reward) <= -0.1 or die else False

        self.frames_queue.append(self.transform(obs))

        return self.create_observation(self.frames_queue), reward, done, info

    def _repeat_step(self, action):
        total_reward = 0
        for _ in range(self.n_repeat_actions):
            if self.render_inside:
                self.env.render()

            obs, reward, die, info = self.env.step(action)
            # green penalty
            if np.mean(obs[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.avg_reward(reward) <= -0.1 else False
            if done or die:
                break

        self.frames_queue.append(self.transform(obs))

        return self.create_observation(self.frames_queue), total_reward, done, info

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env = NormalizedAction(FlattenedAction(env))
    env = CarRacingEnvWrapper(env)
    obs = env.reset()
    print(obs.shape)
    env.close()
