import functools
import random
import torch
import numpy as np
import torch.multiprocessing as mp

from .utils import batch_normalize_images, batch_normalize_grayscale_images, convert_to_CHW_tensor
from .sum_tree import SumTree


MEAN = np.tile([0.3171, 0.3183, 0.3779], 2)
STD = np.tile([0.1406, 0.0594, 0.0925], 2)

OBSERVATION = 0
EXTRA_OBSERVATION = 1
ACTION = 2
REWARD = 3
DONE = 4


__all__ = ['ReplayBuffer', 'EpisodeReplayBuffer', 'EfficientReplayBuffer', 'PrioritizedReplayBuffer']


class ReplayBuffer(object):
    def __init__(self, capacity, initializer, Value=mp.Value, Lock=mp.Lock):
        self.capacity = capacity
        self.buffer = initializer()
        self.buffer_offset = Value('L', 0)
        self.lock = Lock()

    def push(self, *args):
        items = tuple(args)
        with self.lock:
            if self.size < self.capacity:
                self.buffer.append(items)
            else:
                self.buffer[self.offset] = items
            self.offset = (self.offset + 1) % self.capacity

    def extend(self, trajectory):
        with self.lock:
            # trajectory is list of tuples, and this look like wrap tuple with tuple again
            # but this results in the same structure, don't know why use this code
            for items in map(tuple, trajectory):
                if self.size < self.capacity:
                    self.buffer.append(items)
                else:
                    self.buffer[self.offset] = items
                self.offset = (self.offset + 1) % self.capacity

    def sample(self, batch_size):
        batch = []
        for i in np.random.randint(self.size, size=batch_size):
            batch.append(self.buffer[i])

        batch_samples = list(zip(*batch))
        # normalize batch of observations
        batch_samples[0] = batch_normalize_images(batch_samples[0], MEAN, STD)
        batch_samples[4] = batch_normalize_images(batch_samples[4], MEAN, STD)

        return tuple(map(np.stack, batch_samples))

    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.buffer)

    @property
    def offset(self):
        return self.buffer_offset.value

    @offset.setter
    def offset(self, value):
        self.buffer_offset.value = value


class EpisodeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, initializer, Value=mp.Value, Lock=mp.Lock):
        super().__init__(capacity=capacity, initializer=initializer, Value=Value, Lock=Lock)
        # initializer is list
        self.lengths = initializer()
        self.buffer_size = Value('L', 0)
        self.n_total_episodes = Value('L', 0)
        self.length_mean = Value('f', 0.0)
        self.length_square_mean = Value('f', 0.0)

    def push(self, *args):
        # items is whole episode's trajectory
        items = tuple(args)
        length = len(args[0])
        with self.lock:
            buffer_len = len(self.buffer)
            # size is sum of every episode length
            if self.size + length <= self.capacity:
                self.buffer.append(items)
                self.lengths.append(length)
                self.buffer_size.value += length
                # offset tracks len of buffer 1 step behind
                self.offset = buffer_len + 1
            else:
                self.offset %= buffer_len
                self.buffer[self.offset] = items
                self.buffer_size.value += length - self.lengths[self.offset]
                self.lengths[self.offset] = length
                self.offset = (self.offset + 1) % buffer_len
            self.n_total_episodes.value += 1
            self.length_mean.value += (length - self.length_mean.value) \
                                      / self.n_total_episodes.value
            self.length_square_mean.value += (length * length - self.length_square_mean.value) \
                                             / self.n_total_episodes.value

    def sample(self, batch_size, min_length=16):
        ''' Sample entire episode. 1 item in batch is 1 episode '''
        length_mean = self.length_mean.value
        length_square_mean = self.length_square_mean.value
        length_stddev = np.sqrt(length_square_mean - length_mean * length_mean)

        # if std is small (most episodes have similar length)
        if length_stddev / length_mean < 0.1:
            weights = np.ones(shape=(len(self.lengths),))
        else:
            # otherwise, favor long episode
            weights = np.asanyarray(list(self.lengths))
        weights = weights / weights.sum()

        episodes = []
        lengths = []
        for i in range(batch_size):
            while True:
                index = np.random.choice(len(weights), p=weights)

                # size: (length, item_size)
                # observation, action, reward, done
                items = self.buffer[index]
                length = len(items[0])
                if length >= min_length:
                    episodes.append(items)
                    lengths.append(length)
                    break

        return episodes, lengths

    @property
    def size(self):
        return self.buffer_size.value


def modulo_range(start, length, modulo):
  for i in range(length):
    yield (start + i) % modulo


class EfficientReplayBuffer:
    def __init__(self, capacity, batch_size, n_frames, n_step_return,
                 list_initializer, dict_initializer,
                 frame_stack_mode='concatenate', frame_stack_axis=0,
                 gamma=0.99, Value=mp.Value, Lock=mp.Lock, debug=False):
        self.capacity = capacity
        self.buffer = list_initializer()
        self.buffer_offset = Value('L', 0)
        self.lock = Lock()

        assert n_frames > 1
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_step_return = n_step_return
        self.gamma = gamma

        self.next_episode_start = True
        self.end_episode_indexes = dict_initializer()
        self.invalid_indexes = list_initializer()
        self.invalid_indexes.extend([i for i in range(self.n_frames - 1)])
        self.pad_indexes = dict_initializer()

        assert frame_stack_mode in ['stack', 'concatenate']

        if frame_stack_mode == 'stack':
            self.frame_stack_func = functools.partial(np.stack, axis=frame_stack_axis)
        else:
            self.frame_stack_func = functools.partial(np.concatenate, axis=frame_stack_axis)

        self.max_sample_attempt = 10 if debug else 100

    def push(self, *args):
        items = tuple(args)
        with self.lock:
            if self.size < self.capacity:
                self.buffer.append(items)
            else:
                self.buffer[self.offset] = items
            self.offset = (self.offset + 1) % self.capacity

    def extend(self, trajectory, is_end=True):
        with self.lock:
            self._extend(trajectory, is_end)

    def _extend(self, trajectory, is_end):
        if self.next_episode_start:
            self._pad_transition(trajectory[0])
            self.next_episode_start = False

        for items in trajectory:
            if self.size < self.capacity:
                self.buffer.append(items)
            else:
                self.buffer[self.offset] = items

            self.end_episode_indexes.pop(self.offset, None)
            self.pad_indexes.pop(self.offset, None)
            self.offset = (self.offset + 1) % self.capacity

        if is_end:
            self.end_episode_indexes[(self.offset - 1) % self.capacity] = True
            self.next_episode_start = True

        self.invalid_indexes[:] = self._get_invalid_end_indexes()

    def sample(self, *args, normalize=False, device='cpu'):
        with self.lock:
            idx_batch = self._sample_index_batch()

            transitions_batch = self._get_transitions_batch(idx_batch)

            if normalize:
                transitions_batch = self._normalize_transitions_batch(transitions_batch, device)

            return transitions_batch

    def _get_transitions_batch(self, idx_batch):
        batch = []
        for state_idx in idx_batch:
            # this slice doesn't get item at step_idx + n_step_return
            # which is for next obs only
            transitions_indexes = slice(state_idx,
                                        state_idx + self.n_step_return)
            transitions_dones = [t[DONE] for t in self.buffer[transitions_indexes]]
            is_done = any(transitions_dones)

            if is_done:
                # because argmax return index not count, +1 make it count
                transition_len = np.argmax(transitions_dones) + 1
            else:
                transition_len = self.n_step_return

            next_state_idx = (state_idx + transition_len) % self.capacity

            current_transition = self.buffer[state_idx]

            obs = self._get_obs_stack(state_idx)
            extra_obs = current_transition[EXTRA_OBSERVATION]
            action = current_transition[ACTION]
            reward = self._sum_gamma_rewards(state_idx, next_state_idx)
            next_obs = self._get_obs_stack(next_state_idx)
            next_extra_obs = self.buffer[next_state_idx][EXTRA_OBSERVATION]

            batch.append((obs, extra_obs, action, [reward], next_obs, next_extra_obs, [is_done]))

        return tuple(map(np.stack, zip(*batch)))

    def _get_obs_stack(self, idx):
        start_idx = (idx - self.n_frames + 1) % self.capacity
        end_idx = idx + 1

        obses = [transition[OBSERVATION] for transition in self._get_transition_list(start_idx, end_idx)]
        return self.frame_stack_func(obses)

    def _get_done_list(self, idx):
        start_idx = (idx - self.n_frames + 1) % self.capacity
        end_idx = idx + 1

        return [transition[DONE] for transition in self._get_transition_list(start_idx, end_idx)]

    def _get_transition_list(self, start_idx, end_idx):
        ''' return list of transition inclusive start, exclusive ends '''
        if start_idx < end_idx:
            return [transition for transition in self.buffer[start_idx:end_idx]]
        else:
            idx = start_idx
            transitions = []
            while True:
                transitions.append(self.buffer[idx])
                idx = (idx + 1) % self.capacity
                if idx == end_idx:
                    break

            return transitions

    def _is_valid_transition(self, idx):
        if idx < 0 or idx >= self.capacity:
            return False

        if not self.is_full:
            if idx < self.n_frames - 1 or \
                idx > self.offset - self.n_step_return:
                return False

        if idx in self.invalid_indexes:
            return False

        if idx in self.pad_indexes:
            return False

        if any(self._get_done_list(idx)[:-1]):
            return False

        # look forward and accept end of episode, if it was terminal state
        for t in modulo_range(idx, self.n_step_return, self.capacity):
            if t in self.end_episode_indexes and not self.buffer[t][DONE]:
                return False

        return True

    def _get_invalid_end_indexes(self):
        ''' N-step before current position isn't valid since it can't use N-step
            n_frames after current position isn't valid since it can stack frame
            to this position '''
        return [(self.offset - self.n_step_return + i) % self.capacity \
            for i in range(self.n_step_return + self.n_frames)]

    def _sum_gamma_rewards(self, state_idx, next_state_idx):
        if state_idx < next_state_idx:
            rewards = [transition[REWARD] for transition in self.buffer[state_idx:next_state_idx]]
        else:
            rewards = []
            for i in range(self.n_step_return):
                reward_idx = (state_idx + i) % self.capacity
                if reward_idx == next_state_idx:
                    break

                rewards.append(self.buffer[reward_idx][REWARD])

        cumulative_reward = 0
        discount_pow = 0

        for reward in rewards:
            cumulative_reward += reward * self.gamma**discount_pow
            discount_pow += 1

        return cumulative_reward

    def _pad_transition(self, based_transaction):
        ''' Pad by a number of stack frames '''
        self.pad_indexes[self.offset] = True

        pad_transaction = (
            based_transaction[OBSERVATION],
            based_transaction[EXTRA_OBSERVATION],
            based_transaction[ACTION],
            0,
            based_transaction[DONE]
        )

        for _ in range(self.n_frames - 1):
            if self.size < self.capacity:
                self.buffer.append(pad_transaction)
            else:
                self.buffer[self.offset] = pad_transaction

            self.end_episode_indexes.pop(self.offset, None)
            self.offset = (self.offset + 1) % self.capacity

    def _dump_range(self, start, end):
        for i, (obs, extra_obs, action, reward, done) in enumerate(self.buffer[start:end]):
            print(f'No.{start+i}: action type {type(action)} value {action}, reward {reward}, done {done}')
    
    def __len__(self):
        return self.size

    def _create_sample_tensors(self, np_samples, device):
        return tuple(torch.tensor(sample_item, dtype=torch.float32, device=device) for sample_item in np_samples)

    def _sample_index_batch(self):
        idx_batch = []

        attemp_count = 0
        while len(idx_batch) < self.batch_size and attemp_count < self.max_sample_attempt:
            random_idx = random.randint(0, len(self.buffer) - 1)
            if self._is_valid_transition(random_idx):
                idx_batch.append(random_idx)
                attemp_count = 0
            else:
                attemp_count += 1

        if len(idx_batch) != self.batch_size:
            raise RuntimeError('cannot sample enough valid sample')

        return idx_batch

    def _normalize_transitions_batch(self, transitions_batch, device):
        observation, extra_state, action, reward, next_observation, next_extra_state, done = self._create_sample_tensors(transitions_batch, device)

        self._create_normalization_params_tensor(MEAN, STD, device)

        observation, next_observation = self._normalize_observations(observation, next_observation)
        observation = convert_to_CHW_tensor(observation)
        next_observation = convert_to_CHW_tensor(next_observation)

        return observation, extra_state, action, reward, next_observation, next_extra_state, done

    def _create_normalization_params_tensor(self, mean, std, device):
        if not hasattr(self, 'mean_tensor'):
            self.mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
            self.std_tensor = torch.tensor(std, dtype=torch.float32, device=device)

    def _normalize_observations(self, observation, next_observation):
        observation = observation.div_(255.).subtract_(self.mean_tensor).divide_(self.std_tensor)
        next_observation = next_observation.div_(255.).subtract_(self.mean_tensor).divide_(self.std_tensor)

        return observation, next_observation

    @property
    def is_full(self):
        return self.size == self.capacity

    @property
    def size(self):
        return len(self.buffer)

    @property
    def offset(self):
        return self.buffer_offset.value

    @offset.setter
    def offset(self, value):
        self.buffer_offset.value = value


def range_wrapped(start, end, length):
    if start < end:
        return list(range(start, end))

    i = start
    values = []
    while i != end:
        values.append(i)
        i = (i + 1) % length

    return values


class PrioritizedReplayBuffer(EfficientReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        capacity = kwargs['capacity']
        list_initializer = kwargs['list_initializer']
        self.sum_tree = SumTree(capacity=capacity, list_initializer=list_initializer)
        self.lock = self.lock if kwargs.get('debug', False) else mp.RLock()
        self.sampled_indexes = list_initializer()

    def extend(self, trajectory, is_end=True):
        with self.lock:
            start_idx = self.offset
            self._extend(trajectory, is_end)
            end_idx = self.offset

            for i in range_wrapped(start_idx, end_idx, self.capacity):
                self.sum_tree.set(i, self.sum_tree.max_recorded_priority)

    def sample(self, *args, normalize=False, device='cpu'):
        # release in set_priority
        self.lock.acquire()

        idx_batch = self._sample_index_batch()
        self.sampled_indexes[:] = idx_batch

        transitions_batch = self._get_transitions_batch(idx_batch)

        if normalize:
            transitions_batch = self._normalize_transitions_batch(transitions_batch, device)

        sample_probs = torch.tensor([self.sum_tree.get(i) for i in idx_batch], device=device)

        return transitions_batch + (sample_probs,)

    def set_priority(self, priorities):
        priorities = np.asarray(priorities)

        for index, priority in zip(self.sampled_indexes, priorities):
            self.sum_tree.set(index, priority)

        self.sampled_indexes[:] = []

        self.lock.release()

    def _sample_index_batch(self):
        indices = self.sum_tree.stratified_sample(self.batch_size)
        allowed_attempts = self.max_sample_attempt

        for i in range(len(indices)):
            if not self._is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                        format(self.max_sample_attempt, i, self.batch_size))
                index = indices[i]
                while not self._is_valid_transition(index) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self.sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index

        return indices


class EREBuffer(EfficientReplayBuffer):
    def sample(self, nth_update, normalize=False, device='cpu'):
        with self.lock:
            idx_batch = self._sample(nth_update)

            transitions_batch = self._get_transitions_batch(idx_batch)

            if normalize:
                transitions_batch = self._normalize_transitions_batch(transitions_batch, device)

            return transitions_batch

    N_TOTAL_UPDATE = 256

    def _sample_uniform_index_batch(self, nth_update):
        if self.size < self.capacity:
            ordered_indexes = list(range(self.size-1, -1, -1))
        else:
            ordered_indexes = list(range(self.offset-1, -1, -1)) + list(range(self.capacity-1, self.offset-1, -1))

        if self.size < self.capacity * 0.1:
            ck = self.size
        else:
            N = self.capacity
            if self.size < self.capacity:
                N = self.size

            n = 0.995

            # TODO: anneal n
            ck = int(max(N * n ** (nth_update * 1000 / self.N_TOTAL_UPDATE), self.capacity * 0.1))

        ordered_indexes = ordered_indexes[:ck]

        idx_batch = []

        attemp_count = 0
        while len(idx_batch) < self.batch_size and attemp_count < self.max_sample_attempt:
            random_idx = random.choice(ordered_indexes)
            if self._is_valid_transition(random_idx):
                idx_batch.append(random_idx)
                attemp_count = 0
            else:
                attemp_count += 1

        if len(idx_batch) != self.batch_size:
            raise RuntimeError('cannot sample enough valid sample')

        return idx_batch
