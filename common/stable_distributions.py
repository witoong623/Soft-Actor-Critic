import math
import torch
from torch import distributions as pyd
from numbers import Number



"""
distributions used for the policy
"""


class StableNormal(torch.distributions.Normal):
    ''' address 3. normal-fix '''
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return - 0.5 * ((value - self.loc)  / self.scale )**2 - log_scale - math.log(math.sqrt(2 * math.pi))


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1, threshold=20):
        super().__init__(cache_size=cache_size)
        # if I want to match the paper, threshold must be 10, and beta should be -2
        # in implementation, beta is 1 and threshold is 20 though
        self.softplus = torch.nn.Softplus(threshold=threshold)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - self.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, threshold=20, stable=False):
        self.loc = loc
        self.scale = scale

        if stable:
            # address 3.normal-fix
            self.base_dist = StableNormal(loc, scale)
        else:
            self.base_dist = pyd.Normal(loc, scale)

        transforms = [TanhTransform(threshold=threshold)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
