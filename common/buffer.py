import functools
import random
import numpy as np
import torch.multiprocessing as mp

from .utils import batch_normalize_images, batch_normalize_grayscale_images


MEAN = np.tile([0.4652, 0.4417, 0.3799], 2)
STD = np.tile([0.0946, 0.1767, 0.1865], 2)


__all__ = ['ReplayBuffer', 'EpisodeReplayBuffer']


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

        # batch_samples[0] = batch_normalize_grayscale_images(batch_samples[0])
        # batch_samples[4] = batch_normalize_grayscale_images(batch_samples[4])

        # size: (batch_size, item_size)
        # observation, additional_state, action, reward, next_observation, next_additional_state, done
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
                 initializer, frame_stack_mode='concatenate', frame_stack_axis=0,
                 gamma=0.99, Value=mp.Value, Lock=mp.Lock):
        self.capacity = capacity
        self.buffer = initializer()
        self.buffer_offset = Value('L', 0)
        self.lock = Lock()

        assert n_frames > 1
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_step_return = n_step_return
        self.gamma = gamma

        self.end_episode_indexes = set()
        self.invalid_indexes = [i for i in range(self.n_frames - 1)]

        assert frame_stack_mode in ['stack', 'concatenate']

        if frame_stack_mode == 'stack':
            self.frame_stack_func = functools.partial(np.stack, axis=frame_stack_axis)
        else:
            self.frame_stack_func = functools.partial(np.concatenate, axis=frame_stack_axis)

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
            # pad necessary
            first_transition = trajectory[0]
            self._pad_transition((first_transition[0],
                                  first_transition[1],
                                  0,
                                  0,
                                  first_transition[4]))

            # trajectory is list of tuples, and this look like wrap tuple with tuple again
            # but this results in the same structure, don't know why use this code
            for items in map(tuple, trajectory):
                if self.size < self.capacity:
                    self.buffer.append(items)
                else:
                    self.buffer[self.offset] = items

                self.end_episode_indexes.discard(self.offset)
                self.offset = (self.offset + 1) % self.capacity

            self.end_episode_indexes.add((self.offset - 1) % self.capacity)

            self.invalid_indexes = self._get_invalid_range()

    def sample(self, *args, normalize=False):
        idx_batch = []

        attemp_count = 0
        while len(idx_batch) < self.batch_size and attemp_count < 10:
            random_idx = random.randint(0, len(self.buffer) - 1)
            if self._is_valid_transition(random_idx):
                idx_batch.append(random_idx)
                attemp_count = 0
            else:
                attemp_count += 1
        
        if len(idx_batch) != self.batch_size:
            raise RuntimeError('cannot sample enough valid sample')

        batch = []
        for state_idx in idx_batch:
            # this slice doesn't actually get element at step_idx + self.n_step_return
            # which is for next obs only
            transitions_indexes = slice(state_idx,
                                        state_idx + self.n_step_return)
            # get done element from trajectories
            transitions_dones = [t[4] for t in self.buffer[transitions_indexes]]
            is_done = any(transitions_dones)

            if is_done:
                # because argmax return index not count, +1 make it count
                transition_len = np.argmax(transitions_dones) + 1
            else:
                # prevent len is more than remaining trajectory, -1 to convert to base 0 index
                transition_len = self.n_step_return

            next_state_idx = (state_idx + transition_len) % self.capacity

            current_transition = self.buffer[state_idx]

            obs = self._get_obs_stack(state_idx)
            addi_obs = current_transition[1]
            action = current_transition[2]
            reward = self._sum_gamma_rewards(state_idx, next_state_idx)
            next_obs = self._get_obs_stack(next_state_idx)
            next_addi_obs = self.buffer[next_state_idx][1]

            batch.append((obs, addi_obs, action, [reward], next_obs, next_addi_obs, [is_done]))

        if normalize:
            batch_samples = list(zip(*batch))

            batch_samples[0] = batch_normalize_images(batch_samples[0], MEAN, STD)
            batch_samples[4] = batch_normalize_images(batch_samples[4], MEAN, STD)

            batch.append((obs, addi_obs, action, reward, next_obs, next_addi_obs, is_done))

            return tuple(map(np.stack, batch_samples))
        else:
            return tuple(map(np.stack, zip(*batch)))

    def _get_obs_stack(self, idx):
        start_idx = (idx - self.n_frames + 1) % self.capacity
        end_idx = idx + 1

        obses = [transition[0] for transition in self._get_transition_list(start_idx, end_idx)]
        return self.frame_stack_func(obses)

    def _get_done_list(self, idx):
        start_idx = (idx - self.n_frames + 1) % self.capacity
        end_idx = idx + 1

        return [transition[4] for transition in self._get_transition_list(start_idx, end_idx)]

    def _get_transition_list(self, start_idx, end_idx):
        if start_idx % self.capacity < end_idx % self.capacity:
            return [transition for transition in self.buffer[start_idx:end_idx]]
        else:
            return [self.buffer[(start_idx+i) % self.capacity] for i in range(self.n_frames)]

    def _is_valid_transition(self, idx):
        if idx < 0 or idx >= self.capacity:
            return False

        if not self.is_full:
            if idx < self.n_frames - 1 or \
                idx > self.offset - self.n_step_return:
                return False

        if idx in self.invalid_indexes:
            return False

        if any(self._get_done_list(idx)[:-1]):
            return False

        # accept end of episode, if it was terminal state
        for t in modulo_range(idx, self.n_step_return, self.capacity):
            if t in self.end_episode_indexes and not self.buffer[t][-1]:
                return False

        return True

    def _get_invalid_range(self):
        ''' N-step before current position isn't valid since it can't use N-step
            n_frames after current position isn't valid since it can stack frame
            to this position '''
        return [(self.offset - self.n_step_return + i) % self.capacity \
                for i in range(self.n_step_return + self.n_frames)]

    def _sum_gamma_rewards(self, state_idx, next_state_idx):
        if state_idx < next_state_idx:
            rewards = [transition[3] for transition in self.buffer[state_idx:next_state_idx]]
        else:
            rewards = []
            for i in range(self.n_step_return):
                reward_idx = (state_idx + i) % self.capacity
                if reward_idx == next_state_idx:
                    break

                rewards.append(self.buffer[reward_idx][3])

        cumulative_reward = 0
        discount_pow = 0

        for reward in rewards:
            cumulative_reward += reward * self.gamma**discount_pow
            discount_pow += 1

        return cumulative_reward

    def _pad_transition(self, item):
        ''' Pad by a number of stack frames '''
        for _ in range(self.n_frames - 1):
            if self.size < self.capacity:
                self.buffer.append(item)
            else:
                self.buffer[self.offset] = item

        self.end_episode_indexes.discard(self.offset)
        self.offset = (self.offset + 1) % self.capacity
    
    def __len__(self):
        return self.size

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
