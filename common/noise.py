import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional


# from https://github.com/DLR-RM/stable-baselines3/blob/a2e3001598cbccb85c5fc76b099e04304ebc46a0/stable_baselines3/common/noise.py
class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self):
        super(ActionNoise, self).__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return torch.from_numpy(noise)

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


if __name__ == '__main__':
    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0]), sigma=np.array([0.5]))
    print(ou_noise())
