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
        return f'OBS ep {self.episode} step {self.timestep}'


class MockAddiObs(MockObs):
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, MockAddiObs):
            return __o.episode == self.episode and __o.timestep == self.timestep

    def __repr__(self) -> str:
        return f'Addi OBS ep {self.episode} step {self.timestep}'


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
                                   frame_stack_mode='stack',
                                   list_initializer=list,
                                   dict_initializer=dict,
                                   Value=MockValue,
                                   Lock=MagicMock)


class TestReplayBuffer(unittest.TestCase):
    N_FRAMES = 2
    CAPACITY = 30
    BATCH_SIZE = 2
    N_STEP_RETURN = 2
    GAMMA = 0.99
    N_MAX_STEP = 10

    def create_default_test_buffer(self, capacity=CAPACITY, batch_size=BATCH_SIZE,
                                   n_frames=N_FRAMES, n_step=N_STEP_RETURN):
        return DEFAULT_BUFFER(capacity, batch_size, n_frames, n_step)

    def populate_buffer(self, replay_buff, n_ep=2, n_step=10, start_ep_num=1, start_step_num=1, is_done=True, is_end=True):
        eps = []

        for ep in range(start_ep_num, start_ep_num + n_ep):
            trajectories = [] 
            for step in range(start_step_num, start_step_num + n_step - 1):
                reward = step
                # obs, addition obs, action, reward, done
                trajectories.append((MockObs(ep, step), MockAddiObs(ep, step), 0.1 * step, reward, False))

            trajectories.append((MockObs(ep, step + 1), MockAddiObs(ep, step + 1), 0.1 * (step + 1), reward + 1, is_done))

            eps.append(trajectories)

        for ep in eps:
            replay_buff.extend(ep, is_end)

    def assertRewardValid(self, step, actual_reward, n_step_return=N_STEP_RETURN, max_step=N_MAX_STEP):
        ''' calculate expected reward from step '''
        expected_reward = 0
        power = 0
        for step_reward in range(step, min(step + n_step_return, max_step + 1)):
            expected_reward += self.GAMMA**power * step_reward
            power += 1

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

        self.assertEqual(replay_buff.buffer[0][0], MockObs(1, 1))
        self.assertEqual(len(replay_buff.buffer), 11)

        self.assertEqual(replay_buff.invalid_indexes, [9, 10, 11, 12])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

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

        self.assertEqual(get_obs(replay_buff, 0), MockObs(1, 1))
        self.assertEqual(get_obs(replay_buff, 11), MockObs(2, 1))
        self.assertEqual(get_obs(replay_buff, 21), MockObs(2, 10))
        self.assertEqual(get_done(replay_buff, 21), True)
        self.assertEqual(len(replay_buff.buffer), 22)

        self.assertEqual(replay_buff.invalid_indexes, [20, 21, 22, 23])
        self.assertDictEqual(replay_buff.pad_indexes, {0: True, 11: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([20, 21, 22, 23], cycle=True)):
                replay_buff.sample()

    def test_invalid_indexes_add_three_ep(self):
        ''' this is a wraparound case '''
        n_ep = 3
        n_step = 10
        replay_buff = self.create_default_test_buffer()

        self.populate_buffer(replay_buff, n_ep, n_step)

        self.assertEqual(get_obs(replay_buff, 0), MockObs(3, 8))
        self.assertEqual(get_obs(replay_buff, 1), MockObs(3, 9))
        self.assertEqual(get_obs(replay_buff, 2), MockObs(3, 10))
        self.assertEqual(get_obs(replay_buff, 3), MockObs(1, 3))

        self.assertEqual(get_obs(replay_buff, 22), MockObs(3, 1))
        self.assertEqual(get_obs(replay_buff, 23), MockObs(3, 1))
        self.assertEqual(get_obs(replay_buff, 29), MockObs(3, 7))

        with self.assertRaises(IndexError):
            get_obs(replay_buff, 30)

        self.assertEqual(get_done(replay_buff, 2), True)

        self.assertEqual(replay_buff.invalid_indexes, [1, 2, 3, 4])
        self.assertEqual(replay_buff.pad_indexes, {11: True, 22: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([1, 2, 3, 4], cycle=True)):
                replay_buff.sample()

    def test_invalid_indexes_add_one_ep_in_two_steps(self):
        n_ep = 1
        n_step = 5
        replay_buff = self.create_default_test_buffer()
        self.assertEqual(replay_buff.invalid_indexes, [0])

        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [4, 5, 6, 7])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_step_num=6, is_done=True, is_end=True)

        self.assertEqual(replay_buff.buffer[0][0], MockObs(1, 1))
        self.assertEqual(len(replay_buff.buffer), 11)

        self.assertEqual(replay_buff.invalid_indexes, [9, 10, 11, 12])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([9, 10, 11, 12], cycle=True)):
                replay_buff.sample()

    def test_invalid_indexes_add_two_ep_in_two_steps(self):
        n_ep = 1
        n_step = 5
        replay_buff = self.create_default_test_buffer()
        self.assertEqual(replay_buff.invalid_indexes, [0])

        # first ep
        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [4, 5, 6, 7])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_step_num=6, is_done=False, is_end=True)

        self.assertEqual(get_obs(replay_buff, 0), MockObs(1, 1))
        self.assertEqual(len(replay_buff.buffer), 11)

        self.assertEqual(replay_buff.invalid_indexes, [9, 10, 11, 12])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([9, 10, 11, 12], cycle=True)):
                replay_buff.sample()

        # second ep
        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=2, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [15, 16, 17, 18])
        self.assertEqual(replay_buff.pad_indexes, {0: True, 11: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=2, start_step_num=6, is_done=True, is_end=True)

        self.assertEqual(get_obs(replay_buff, 11), MockObs(2, 1))
        self.assertEqual(len(replay_buff.buffer), 22)

        self.assertEqual(replay_buff.invalid_indexes, [20, 21, 22, 23])
        self.assertEqual(replay_buff.pad_indexes, {0: True, 11: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([20, 21, 22, 23, 0, 11], cycle=True)):
                replay_buff.sample()

    def test_invalid_indexes_add_three_ep_in_two_steps(self):
        ''' this is a wraparound case '''
        n_ep = 1
        n_step = 5
        replay_buff = self.create_default_test_buffer()
        self.assertEqual(replay_buff.invalid_indexes, [0])

        # first ep
        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [4, 5, 6, 7])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_step_num=6, is_done=False, is_end=True)

        self.assertEqual(get_obs(replay_buff, 0), MockObs(1, 1))
        self.assertEqual(len(replay_buff.buffer), 11)

        self.assertEqual(replay_buff.invalid_indexes, [9, 10, 11, 12])
        self.assertEqual(replay_buff.pad_indexes, {0: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([9, 10, 11, 12], cycle=True)):
                replay_buff.sample()

        # second ep
        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=2, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [15, 16, 17, 18])
        self.assertEqual(replay_buff.pad_indexes, {0: True, 11: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=2, start_step_num=6, is_done=True, is_end=True)

        self.assertEqual(get_obs(replay_buff, 11), MockObs(2, 1))
        self.assertEqual(len(replay_buff.buffer), 22)

        self.assertEqual(replay_buff.invalid_indexes, [20, 21, 22, 23])
        self.assertEqual(replay_buff.pad_indexes, {0: True, 11: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([20, 21, 22, 23, 0, 11], cycle=True)):
                replay_buff.sample()

        # third ep
        # first step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=3, is_done=False, is_end=False)

        self.assertEqual(replay_buff.invalid_indexes, [26, 27, 28, 29])
        self.assertEqual(replay_buff.pad_indexes, {0: True, 11: True, 22: True})

        # second step
        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=3, start_step_num=6, is_done=True, is_end=True)

        self.assertEqual(get_obs(replay_buff, 22), MockObs(3, 1))
        self.assertEqual(len(replay_buff.buffer), 30)

        self.assertEqual(replay_buff.invalid_indexes, [1, 2, 3, 4])
        self.assertEqual(replay_buff.pad_indexes, {11: True, 22: True})

        with self.assertRaises(RuntimeError):
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([1, 2, 3, 4, 11, 22], cycle=True)):
                replay_buff.sample()

    def test_sample_into_pad(self):
        ''' end of episode has done = true '''
        n_ep = 3
        n_step = 10
        replay_buff = self.create_default_test_buffer()

        self.populate_buffer(replay_buff, n_ep, n_step)

        with self.assertRaises(RuntimeError) as assert_err:
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([11, 22], cycle=True)):
                replay_buff.sample()

        self.assertEqual(str(assert_err.exception), 'cannot sample enough valid sample')

    def test_sample_into_pad_no_terminal(self):
        ''' end of episode has done = false '''
        n_ep = 1
        n_step = 10
        replay_buff = self.create_default_test_buffer()

        self.populate_buffer(replay_buff, n_ep, n_step, is_done=False)

        self.populate_buffer(replay_buff, n_ep, n_step, start_ep_num=2, is_done=True)

        with self.assertRaises(RuntimeError) as assert_err:
            with unittest.mock.patch('random.randint',
                                     get_mock_random_func([11, 22], cycle=True)):
                replay_buff.sample()

    def test_sample_after_add_one_ep(self):
        ''' sample at the beginning and end of episode '''
        n_ep = 1
        n_step = 10
        replay_buff = self.create_default_test_buffer()
        self.populate_buffer(replay_buff, n_ep, n_step)

        # only 1 and 8 are valid
        with unittest.mock.patch('random.randint', get_mock_random_func([0, 1, 11, 10, 9, 8])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

            # assert first transition in batch
            self.assertEqual(obs[0].tolist(), [MockObs(1, 1)]*2)
            self.assertEqual(addi_obs[0], MockAddiObs(1, 1))
            self.assertEqual(action[0], 0.1)
            self.assertRewardValid(1, reward[0])
            self.assertEqual(next_obs[0].tolist(), [MockObs(1, 2), MockObs(1, 3)])
            self.assertEqual(next_addi_obs[0], MockAddiObs(1, 3))
            self.assertFalse(done[0])

            # assert second transition in batch
            self.assertEqual(obs[1].tolist(), [MockObs(1, 7), MockObs(1, 8)])
            self.assertEqual(addi_obs[1], MockAddiObs(1, 8))
            self.assertEqual(action[1], 0.8)
            self.assertRewardValid(8, reward[1])
            self.assertEqual(next_obs[1].tolist(), [MockObs(1, 9), MockObs(1, 10)])
            self.assertEqual(next_addi_obs[1], MockAddiObs(1, 10))
            # done
            # THIS IS FALSE because even though we use next state idx = state idx + n-step
            # that is just next state, the real variable that determine whether it ends
            # at the current t or not is still done of the last transition we use its reward for N-step
            self.assertFalse(done[1])

    def test_sample_after_add_two_eps(self):
        ''' sample at the beginning and end of episodes '''
        n_ep = 2
        n_step = 10
        replay_buff = self.create_default_test_buffer(batch_size=4)
        self.populate_buffer(replay_buff, n_ep, n_step)

        with unittest.mock.patch('random.randint', get_mock_random_func([1, 9, 12, 19])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

            # assert first transition in batch - ep 1, index 1
            self.assertEqual(obs[0].tolist(), [MockObs(1, 1)]*2)
            self.assertEqual(addi_obs[0], MockAddiObs(1, 1))
            self.assertEqual(action[0], 0.1)
            self.assertRewardValid(1, reward[0])
            self.assertEqual(next_obs[0].tolist(), [MockObs(1, 2), MockObs(1, 3)])
            self.assertEqual(next_addi_obs[0], MockAddiObs(1, 3))
            self.assertFalse(done[0])

            # assert second transition in batch - ep 1, index 9
            self.assertEqual(obs[1].tolist(), [MockObs(1, 8), MockObs(1, 9)])
            self.assertEqual(addi_obs[1], MockAddiObs(1, 9))
            self.assertEqual(action[1], 0.9)
            self.assertRewardValid(9, reward[1])
            self.assertEqual(next_obs[1].tolist(), [MockObs(1, 10), MockObs(2, 1)])
            self.assertEqual(next_addi_obs[1], MockAddiObs(2, 1))
            self.assertTrue(done[1])

            # assert third transition in batch - ep 2
            self.assertEqual(obs[2].tolist(), [MockObs(2, 1)]*2)
            self.assertEqual(addi_obs[2], MockAddiObs(2, 1))
            self.assertEqual(action[2], 0.1)
            self.assertRewardValid(1, reward[2])
            self.assertEqual(next_obs[2].tolist(), [MockObs(2, 2), MockObs(2, 3)])
            self.assertEqual(next_addi_obs[2], MockAddiObs(2, 3))
            self.assertFalse(done[2])

            # assert fourth transition in batch
            self.assertEqual(obs[3].tolist(), [MockObs(2, 7), MockObs(2, 8)])
            self.assertEqual(addi_obs[3], MockAddiObs(2, 8))
            self.assertEqual(action[3], 0.8)
            self.assertRewardValid(8, reward[3])
            self.assertEqual(next_obs[3].tolist(), [MockObs(2, 9), MockObs(2, 10)])
            self.assertEqual(next_addi_obs[3], MockAddiObs(2, 10))
            self.assertFalse(done[3])

    def test_sample_after_add_three_eps(self):
        ''' sample at the beginning and end of episodes '''
        n_ep = 3
        n_step = 10
        replay_buff = self.create_default_test_buffer(batch_size=6)
        self.populate_buffer(replay_buff, n_ep, n_step)

        # only 5, 20, 21, 28, 29, 0
        with unittest.mock.patch('random.randint', get_mock_random_func([4, 5, 20, 21, 28, 29, 0])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

            # assert first transition in batch - ep 1, index 5
            self.assertEqual(obs[0].tolist(), [MockObs(1, 4), MockObs(1, 5)])
            self.assertEqual(addi_obs[0], MockAddiObs(1, 5))
            self.assertEqual(action[0], 0.5)
            self.assertRewardValid(5, reward[0])
            self.assertEqual(next_obs[0].tolist(), [MockObs(1, 6), MockObs(1, 7)])
            self.assertEqual(next_addi_obs[0], MockAddiObs(1, 7))
            self.assertFalse(done[0])

            # assert second transition in batch - ep 2, index 20
            self.assertEqual(obs[1].tolist(), [MockObs(2, 8), MockObs(2, 9)])
            self.assertEqual(addi_obs[1], MockAddiObs(2, 9))
            self.assertEqual(action[1], 0.9)
            self.assertRewardValid(9, reward[1])
            self.assertEqual(next_obs[1].tolist(), [MockObs(2, 10), MockObs(3, 1)])
            self.assertEqual(next_addi_obs[1], MockAddiObs(3, 1))
            self.assertTrue(done[1])

            # assert second transition in batch - ep 2, index 21
            self.assertEqual(obs[2].tolist(), [MockObs(2, 9), MockObs(2, 10)])
            self.assertEqual(addi_obs[2], MockAddiObs(2, 10))
            self.assertEqual(action[2], 1)
            self.assertRewardValid(10, reward[2])
            # first element is 2-10 because next state after step 10 (and done) is 3-1
            # it's technically incorrect but it doesn't matter
            # because we don't use it for gredient anyway
            self.assertEqual(next_obs[2].tolist(), [MockObs(2, 10), MockObs(3, 1)])
            self.assertEqual(next_addi_obs[2], MockAddiObs(3, 1))
            self.assertTrue(done[2])

            # assert second transition in batch - ep 3, index 28 (step 6)
            self.assertEqual(obs[3].tolist(), [MockObs(3, 5), MockObs(3, 6)])
            self.assertEqual(addi_obs[3], MockAddiObs(3, 6))
            self.assertAlmostEqual(action[3], 0.6)
            self.assertRewardValid(6, reward[3])
            self.assertEqual(next_obs[3].tolist(), [MockObs(3, 7), MockObs(3, 8)])
            self.assertEqual(next_addi_obs[3], MockAddiObs(3, 8))
            self.assertFalse(done[3])

            # assert second transition in batch - ep 3, index 29 (step 7)
            self.assertEqual(obs[4].tolist(), [MockObs(3, 6), MockObs(3, 7)])
            self.assertEqual(addi_obs[4], MockAddiObs(3, 7))
            self.assertAlmostEqual(action[4], 0.7)
            self.assertRewardValid(7, reward[4])
            self.assertEqual(next_obs[4].tolist(), [MockObs(3, 8), MockObs(3, 9)])
            self.assertEqual(next_addi_obs[4], MockAddiObs(3, 9))
            self.assertFalse(done[4])

            # assert second transition in batch - ep 3, index 0 (step 8)
            self.assertEqual(obs[5].tolist(), [MockObs(3, 7), MockObs(3, 8)])
            self.assertEqual(addi_obs[5], MockAddiObs(3, 8))
            self.assertEqual(action[5], 0.8)
            self.assertRewardValid(8, reward[5])
            self.assertEqual(next_obs[5].tolist(), [MockObs(3, 9), MockObs(3, 10)])
            self.assertEqual(next_addi_obs[5], MockAddiObs(3, 10))
            self.assertFalse(done[5])

    def test_sample_after_add_one_ep_in_two_steps(self):
        ''' sample at the beginning and end of episodes '''
        n_ep = 1
        n_step = 5
        replay_buff = self.create_default_test_buffer(batch_size=3)

        # first time
        self.populate_buffer(replay_buff, n_ep, n_step, start_step_num=1, is_done=False, is_end=False)
        # second time
        self.populate_buffer(replay_buff, n_ep, n_step, start_step_num=6, is_done=True, is_end=True)

        with unittest.mock.patch('random.randint', get_mock_random_func([1, 6, 8])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

        self.assertEqual(obs[0].tolist(), [MockObs(1, 1)]*2)
        self.assertEqual(addi_obs[0], MockAddiObs(1, 1))
        self.assertAlmostEqual(action[0], 0.1)
        self.assertRewardValid(1, reward[0])
        self.assertEqual(next_obs[0].tolist(), [MockObs(1, 2), MockObs(1, 3)])
        self.assertEqual(next_addi_obs[0], MockAddiObs(1, 3))
        self.assertFalse(done[0])

        self.assertEqual(obs[1].tolist(), [MockObs(1, 5), MockObs(1, 6)])
        self.assertEqual(addi_obs[1], MockAddiObs(1, 6))
        self.assertAlmostEqual(action[1], 0.6)
        self.assertRewardValid(6, reward[1])
        self.assertEqual(next_obs[1].tolist(), [MockObs(1, 7), MockObs(1, 8)])
        self.assertEqual(next_addi_obs[1], MockAddiObs(1, 8))
        self.assertFalse(done[1])

        self.assertEqual(obs[2].tolist(), [MockObs(1, 7), MockObs(1, 8)])
        self.assertEqual(addi_obs[2], MockAddiObs(1, 8))
        self.assertAlmostEqual(action[2], 0.8)
        self.assertRewardValid(8, reward[2])
        self.assertEqual(next_obs[2].tolist(), [MockObs(1, 9), MockObs(1, 10)])
        self.assertEqual(next_addi_obs[2], MockAddiObs(1, 10))

    def test_sample_after_add_two_eps_in_two_steps(self):
        ''' sample at the beginning and end of episodes '''
        replay_buff = self.create_default_test_buffer(batch_size=4)

        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=1, start_step_num=1, is_done=False, is_end=False)
        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=1, start_step_num=6, is_done=True, is_end=True)

        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=2, start_step_num=1, is_done=False, is_end=False)
        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=2, start_step_num=6, is_done=True, is_end=True)

        with unittest.mock.patch('random.randint', get_mock_random_func([1, 9, 12, 19])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

            # assert first transition in batch - ep 1, index 1
            self.assertEqual(obs[0].tolist(), [MockObs(1, 1)]*2)
            self.assertEqual(addi_obs[0], MockAddiObs(1, 1))
            self.assertEqual(action[0], 0.1)
            self.assertRewardValid(1, reward[0])
            self.assertEqual(next_obs[0].tolist(), [MockObs(1, 2), MockObs(1, 3)])
            self.assertEqual(next_addi_obs[0], MockAddiObs(1, 3))
            self.assertFalse(done[0])

            # assert second transition in batch - ep 1, index 9
            self.assertEqual(obs[1].tolist(), [MockObs(1, 8), MockObs(1, 9)])
            self.assertEqual(addi_obs[1], MockAddiObs(1, 9))
            self.assertEqual(action[1], 0.9)
            self.assertRewardValid(9, reward[1])
            self.assertEqual(next_obs[1].tolist(), [MockObs(1, 10), MockObs(2, 1)])
            self.assertEqual(next_addi_obs[1], MockAddiObs(2, 1))
            self.assertTrue(done[1])

            # assert third transition in batch - ep 2
            self.assertEqual(obs[2].tolist(), [MockObs(2, 1)]*2)
            self.assertEqual(addi_obs[2], MockAddiObs(2, 1))
            self.assertEqual(action[2], 0.1)
            self.assertRewardValid(1, reward[2])
            self.assertEqual(next_obs[2].tolist(), [MockObs(2, 2), MockObs(2, 3)])
            self.assertEqual(next_addi_obs[2], MockAddiObs(2, 3))
            self.assertFalse(done[2])

            # assert fourth transition in batch
            self.assertEqual(obs[3].tolist(), [MockObs(2, 7), MockObs(2, 8)])
            self.assertEqual(addi_obs[3], MockAddiObs(2, 8))
            self.assertEqual(action[3], 0.8)
            self.assertRewardValid(8, reward[3])
            self.assertEqual(next_obs[3].tolist(), [MockObs(2, 9), MockObs(2, 10)])
            self.assertEqual(next_addi_obs[3], MockAddiObs(2, 10))
            self.assertFalse(done[3])

    def test_sample_after_add_three_eps_in_two_steps(self):
        ''' sample at the beginning and end of episodes '''
        replay_buff = self.create_default_test_buffer(batch_size=6)

        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=1, start_step_num=1, is_done=False, is_end=False)
        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=1, start_step_num=6, is_done=True, is_end=True)

        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=2, start_step_num=1, is_done=False, is_end=False)
        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=2, start_step_num=6, is_done=True, is_end=True)

        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=3, start_step_num=1, is_done=False, is_end=False)
        self.populate_buffer(replay_buff, n_ep=1, n_step=5,
                             start_ep_num=3, start_step_num=6, is_done=True, is_end=True)

        # only 5, 20, 21, 28, 29, 0
        with unittest.mock.patch('random.randint', get_mock_random_func([4, 5, 20, 21, 28, 29, 0])):
            obs, addi_obs, action, reward, next_obs, next_addi_obs, done = replay_buff.sample()

            # assert first transition in batch - ep 1, index 5
            self.assertEqual(obs[0].tolist(), [MockObs(1, 4), MockObs(1, 5)])
            self.assertEqual(addi_obs[0], MockAddiObs(1, 5))
            self.assertEqual(action[0], 0.5)
            self.assertRewardValid(5, reward[0])
            self.assertEqual(next_obs[0].tolist(), [MockObs(1, 6), MockObs(1, 7)])
            self.assertEqual(next_addi_obs[0], MockAddiObs(1, 7))
            self.assertFalse(done[0])

            # assert second transition in batch - ep 2, index 20
            self.assertEqual(obs[1].tolist(), [MockObs(2, 8), MockObs(2, 9)])
            self.assertEqual(addi_obs[1], MockAddiObs(2, 9))
            self.assertEqual(action[1], 0.9)
            self.assertRewardValid(9, reward[1])
            self.assertEqual(next_obs[1].tolist(), [MockObs(2, 10), MockObs(3, 1)])
            self.assertEqual(next_addi_obs[1], MockAddiObs(3, 1))
            self.assertTrue(done[1])

            # assert second transition in batch - ep 2, index 21
            self.assertEqual(obs[2].tolist(), [MockObs(2, 9), MockObs(2, 10)])
            self.assertEqual(addi_obs[2], MockAddiObs(2, 10))
            self.assertEqual(action[2], 1)
            self.assertRewardValid(10, reward[2])
            # first element is 2-10 because next state after step 10 (and done) is 3-1
            # it's technically incorrect but it doesn't matter
            # because we don't use it for gredient anyway
            self.assertEqual(next_obs[2].tolist(), [MockObs(2, 10), MockObs(3, 1)])
            self.assertEqual(next_addi_obs[2], MockAddiObs(3, 1))
            self.assertTrue(done[2])

            # assert second transition in batch - ep 3, index 28 (step 6)
            self.assertEqual(obs[3].tolist(), [MockObs(3, 5), MockObs(3, 6)])
            self.assertEqual(addi_obs[3], MockAddiObs(3, 6))
            self.assertAlmostEqual(action[3], 0.6)
            self.assertRewardValid(6, reward[3])
            self.assertEqual(next_obs[3].tolist(), [MockObs(3, 7), MockObs(3, 8)])
            self.assertEqual(next_addi_obs[3], MockAddiObs(3, 8))
            self.assertFalse(done[3])

            # assert second transition in batch - ep 3, index 29 (step 7)
            self.assertEqual(obs[4].tolist(), [MockObs(3, 6), MockObs(3, 7)])
            self.assertEqual(addi_obs[4], MockAddiObs(3, 7))
            self.assertAlmostEqual(action[4], 0.7)
            self.assertRewardValid(7, reward[4])
            self.assertEqual(next_obs[4].tolist(), [MockObs(3, 8), MockObs(3, 9)])
            self.assertEqual(next_addi_obs[4], MockAddiObs(3, 9))
            self.assertFalse(done[4])

            # assert second transition in batch - ep 3, index 0 (step 8)
            self.assertEqual(obs[5].tolist(), [MockObs(3, 7), MockObs(3, 8)])
            self.assertEqual(addi_obs[5], MockAddiObs(3, 8))
            self.assertEqual(action[5], 0.8)
            self.assertRewardValid(8, reward[5])
            self.assertEqual(next_obs[5].tolist(), [MockObs(3, 9), MockObs(3, 10)])
            self.assertEqual(next_addi_obs[5], MockAddiObs(3, 10))
            self.assertFalse(done[5])
