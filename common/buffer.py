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
