import functools
import itertools
import unittest
import unittest.mock

from unittest.mock import Mock, MagicMock
from common.buffer import EfficientReplayBuffer


class MockObs:
    def __init__(self, episode, timestep):
        self.episode = episode
        self.timestep = timestep

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, MockObs):
            return __o.episode == self.episode and __o.timestep == self.timestep

    def __repr__(self) -> str:
        return f'EP:{self.episode}--t:{self.timestep}'


class MockValue:
    def __init__(self, _, value) -> None:
        self.value = value


def get_obs(replay_buffer, index):
    return replay_buffer.buffer[index][0]


def get_done(replay_buffer, index):
    return replay_buffer.buffer[index][-1]


def get_mock_random_func(nums_to_random, cycle=False):
    if cycle:
        nums_iter = itertools.cycle(nums_to_random)
    else:
        nums_iter = iter(nums_to_random)

    def mock_random_func(*args, **kwargs):
        start, stop = args
        retry_count = 0
        while retry_count < 10:
            next_val = next(nums_iter)
            if start <= next_val <= stop:
                retry_count = 0
                return next_val
            else:
                retry_count += 1
        
        raise ValueError('nums_to_random are invalid of function being test')

    return mock_random_func


DEFAULT_BUFFER = functools.partial(EfficientReplayBuffer,
                                   initializer=list,
                                   Value=MockValue,
                                   Lock=MagicMock)


class TestReplayBuffer(unittest.TestCase):
    N_FRAMES = 2
    CAPACITY = 30
    BATCH_SIZE = 2
    N_STEP = 2
    GAMMA = 0.99

    def create_default_test_buffer(self, capacity=CAPACITY, batch_size=BATCH_SIZE,
                                   n_frames=N_FRAMES, n_step=N_STEP):
        return DEFAULT_BUFFER(capacity, batch_size, n_frames, n_step)

    def populate_buffer(self, replay_buff, n_ep=2, n_step=10):
        eps = []

        for ep in range(1, n_ep+1):
            trajectories = []
            for step in range(1, n_step):
                reward = step
                # obs, addition obs, action, reward, done
                trajectories.append((MockObs(ep, step), MockObs(ep, step), 0.1, reward, False))

            # terminal transition
            trajectories.append((MockObs(ep, n_step), MockObs(ep, n_step), 0.1, reward, True))

            eps.append(trajectories)

        for ep in eps:
            replay_buff.extend(ep)
    
    def assertRewardValid(self, step, actual_reward, n_step_return=N_STEP):
        ''' calculate expected reward from step '''
        expected_reward = 0
        current_timestep_reward = step
        for i in range(n_step_return):
            expected_reward += self.GAMMA**i * current_timestep_reward
            # reward increase by 1 every step
            current_timestep_reward += 1

        assert expected_reward == actual_reward, f'{expected_reward} != {actual_reward}'

    def test_add_transition(self):
        n_ep = 2
        n_step = 10
        replay_buff = self.create_default_test_buffer()
        self.populate_buffer(replay_buff, n_ep, n_step)

        offset = 0

        for ep in range(1, n_ep+1):
            # asssert pad
            self.assertEqual(get_obs(replay_buff, offset), MockObs(ep, 1))
            offset = ((offset + 1) % self.CAPACITY)

            for step in range(1, n_step+1):
                self.assertEqual(get_obs(replay_buff, offset), MockObs(ep, step))
                offset = ((offset + 1) % self.CAPACITY)

    def test_invalid_indexes_add_one_ep(self):
        n_ep = 1
        n_step = 10
        replay_buff = self.create_default_test_buffer()
        self.assertEqual(replay_buff.invalid_indexes, [0])

        self.populate_buffer(replay_buff, n_ep, n_step)

        # assert pad transition len at the beginning
        self.assertEqual(len(replay_buff.buffer[0]), 1)
        self.assertEqual(replay_buff.buffer[0][0], MockObs(1, 1))
        self.assertEqual(len(replay_buff.buffer), 11)

        self.assertEqual(replay_buff.invalid_indexes, [9, 10, 11, 12])

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([9, 10, 11, 12], cycle=True)):
                replay_buff.sample()

    def test_invalid_indexes_add_two_ep(self):
        n_ep = 2
        n_step = 10
        replay_buff = self.create_default_test_buffer()
        self.assertEqual(replay_buff.invalid_indexes, [0])

        self.populate_buffer(replay_buff, n_ep, n_step)

        # assert pad transition len at the beginning
        self.assertEqual(len(replay_buff.buffer[0]), 1)
        self.assertEqual(get_obs(replay_buff, 0), MockObs(1, 1))
        self.assertEqual(get_obs(replay_buff, 11), MockObs(2, 1))
        self.assertEqual(get_obs(replay_buff, 21), MockObs(2, 10))
        self.assertEqual(get_done(replay_buff, 21), True)
        self.assertEqual(len(replay_buff.buffer), 22)


        self.assertEqual(replay_buff.invalid_indexes, [20, 21, 22, 23])

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([20, 21, 22, 23], cycle=True)):
                replay_buff.sample()

    def test_sample_after_add_one_ep(self):
        ''' start at the beginning, stack should contains repaet obs '''
        n_ep = 1
        n_step = 10
        replay_buff = self.create_default_test_buffer()
        self.populate_buffer(replay_buff, n_ep, n_step)

        # only 1 and 8 are valid
        with unittest.mock.patch('random.randint', get_mock_random_func([0, 1, 11, 10, 9, 8])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()
            # assert first obs in batch
            self.assertEqual(obs[0, 0], MockObs(1, 1))
            self.assertEqual(obs[0, 1], MockObs(1, 1))
            # assert first reward in batch
            self.assertRewardValid(1, reward[0])

            # assert second obs in batch
            self.assertEqual(obs[1][0], MockObs(1, 7))
            self.assertEqual(obs[1][1], MockObs(1, 8))
