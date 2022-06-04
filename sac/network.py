import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from common.network import MultilayerPerceptron, VAEBase, BottleneckNeuralNetwork
from common.networkbase import Container
from common.utils import encode_vae_observation, clone_network
from common.stable_distributions import SquashedNormal


__all__ = [
    'StateEncoderWrapper',
    'Actor', 'Critic'
]

LOG_STD_MIN = np.log(1E-8)
LOG_STD_MAX = np.log(20.0)


class StateEncoderWrapper(Container):
    def __init__(self, encoder, device=None):
        super().__init__()

        self.encoder = encoder

        self.to(device)

        self.reset()

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)

    @torch.no_grad()
    def encode(self, observation, return_tensor=False, data_dtype=torch.float32):
        if isinstance(self.encoder, VAEBase):
            observation = np.expand_dims(observation, axis=0)
            encoded = encode_vae_observation(observation, self.encoder, device=self.device, normalize=False, output_device=self.device)
        elif isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=data_dtype, device=self.device).unsqueeze(dim=0)
            encoded = self(observation)

        if return_tensor:
            encoded = encoded.squeeze(dim=0)
        else:
            encoded = encoded.cpu()
            # if encoded.dtype is not torch.float16:
            #     encoded = encoded.half()
            encoded = encoded.numpy()[0]

        return encoded

    def reset(self):
        pass

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)


class SeparatedStateEncoderWrapper(Container):
    def __init__(self, encoder, device=None, actor_only=False):
        super().__init__()

        self.actor_encoder = encoder
        if not actor_only:
            self.critic_encoder = clone_network(encoder, device)

        self.actor_only = actor_only

        self.to(device)

        self.reset()

    def forward(self, *input, **kwargs):
        return self.actor_encoder(*input, **kwargs), self.critic_encoder(*input, **kwargs)

    @torch.no_grad()
    def encode(self, observation, return_tensor=False):
        if isinstance(self.encoder, VAEBase):
            observation = np.expand_dims(observation, axis=0)
            encoded = encode_vae_observation(observation, self.actor_encoder, device=self.device, normalize=False)
        elif isinstance(observation, np.ndarray):
            obs_dtype = torch.float16 if observation.dtype == np.float16 else torch.float32
            observation = torch.tensor(observation, dtype=obs_dtype, device=self.device).unsqueeze(dim=0)
            encoded = self.actor_encoder(observation)

        if return_tensor:
            encoded = encoded.squeeze(dim=0)
        else:
            encoded = encoded.cpu()
            # if encoded.dtype is not torch.float16:
            #     encoded = encoded.half()
            encoded = encoded.numpy()[0]

        return encoded

    def reset(self):
        pass

    @property
    def encoder(self):
        return self.actor_encoder

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.actor_encoder, name)


class DimensionScaler(Container):
    def __init__(self, input_dim, output_dim, device=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.scaler = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        nn.init.zeros_(self.scaler.weight)
        nn.init.zeros_(self.scaler.bias)
        with torch.no_grad():
            for _, i, o in zip(range(max(input_dim, output_dim)),
                               itertools.cycle(range(input_dim)),
                               itertools.cycle(range(output_dim))):
                self.scaler.weight[o, i] = 1.0

        self.to(device)

    def forward(self, action):
        return self.scaler(action)

    def plot(self):
        input_dim = self.input_dim
        output_dim = self.output_dim

        weight = self.scaler.weight.detach().cpu()
        if weight.dtype is not torch.float16:
            weight = weight.half()
        weight = weight.numpy()

        bias = self.scaler.bias.detach().cpu()
        if bias.dtype is not torch.float16:
            bias = bias.half()
        bias = bias.numpy()
        
        bias = np.expand_dims(bias, axis=1)

        weight = np.concatenate([weight, np.zeros_like(bias), bias], axis=1)

        vmax = np.abs(weight).max()

        fig, ax = plt.subplots(figsize=(input_dim + 3.5, output_dim))

        im = ax.imshow(weight, origin='upper', aspect='equal', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        fig.colorbar(im, ax=ax)
        for (i, j), v in np.ndenumerate(weight):
            if j != input_dim:
                ax.text(j, i, s=f'{v:.2f}',
                        horizontalalignment='center',
                        verticalalignment='center')

        for j in range(output_dim + 1):
            alpha = 0.5
            linestyle = ':'
            if j == 0 or j == output_dim:
                alpha = 1.0
                linestyle = '-'
            ax.axhline(j - 0.5, xmin=0.05 / (input_dim + 2.1), xmax=1.0 - 2.05 / (input_dim + 2.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
            ax.axhline(j - 0.5, xmin=1.0 - 1.05 / (input_dim + 2.1), xmax=1.0 - 0.05 / (input_dim + 2.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
        for i in range(input_dim + 1):
            alpha = 0.5
            linestyle = ':'
            if i == 0 or i == input_dim:
                alpha = 1.0
                linestyle = '-'
            ax.axvline(i - 0.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
        ax.add_artist(plt.Rectangle(xy=(input_dim - 0.5, -0.5),
                                    width=1, height=output_dim,
                                    color=fig.get_facecolor()))
        ax.axvline(input_dim + 0.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                   color='black', linestyle='-', linewidth=1.0)
        ax.axvline(input_dim + 1.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                   color='black', linestyle='-', linewidth=1.0)

        ax.tick_params(top=False, bottom=False, left=False, right=False)

        ax.set_xlim(left=-0.55, right=input_dim + 1.55)
        ax.set_ylim(top=-0.55, bottom=output_dim - 0.45)

        ax.set_xticks(ticks=[(input_dim - 1) / 2, input_dim + 1])
        ax.set_xticklabels(labels=['$w$', '$b$'])
        ax.set_yticks(ticks=[])

        for spline in ax.spines.values():
            spline.set_visible(False)

        fig.tight_layout()
        return fig


class ValueNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, hidden_dims, activation=nn.ReLU(inplace=True), device=None):
        super().__init__(n_dims=[state_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None)

        self.state_dim = state_dim

        self.to(device)

    def forward(self, state):
        return super().forward(state)


class SoftQNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=nn.ReLU(inplace=True), device=None):
        scaled_action_dim = max(state_dim, action_dim)

        super().__init__(n_dims=[state_dim + scaled_action_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scaled_action_dim = scaled_action_dim

        self.action_scaler = DimensionScaler(input_dim=action_dim,
                                             output_dim=scaled_action_dim)

        self.to(device)

    def forward(self, state, action):
        return super().forward(torch.cat([state, self.action_scaler(action)], dim=-1))


class PolicyNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=nn.ReLU(inplace=True), device=None,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        # the last layer of perception is the action_dim (output size) * 2, which can be chunk into 2 action_dims
        super().__init__(n_dims=[state_dim, *hidden_dims, 2 * action_dim],
                         activation=activation,
                         output_activation=None)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.to(device)

    def forward(self, state):
        mean, log_std = super().forward(state).chunk(chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def evaluate(self, state, epsilon=1E-6):
        mean, std = self(state)

        # enforcing Action Bounds
        distribution = Normal(mean, std)
        u = distribution.rsample()
        action = torch.tanh(u)
        # equation 26
        log_prob = distribution.log_prob(u) - torch.log(1.0 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, distribution

    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float16, device=self.device).unsqueeze(0)
        else:
            # Tensor
            state = state.unsqueeze(dim=0).to(self.device)
        mean, std = self(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            z = Normal(0, 1).sample()
            action = torch.tanh(mean + std * z)

        action = action.cpu()
        # if action.dtype is not torch.float16:
        #     action = action.half()
        action = action.numpy()[0]
        return action


class StablePolicyNetwork(PolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=nn.ReLU(inplace=True), device=None, log_std_min=-5, log_std_max=2):
        super().__init__(state_dim, action_dim, hidden_dims, activation, device, log_std_min, log_std_max)

    def forward(self, state):
        mean, log_std = super().forward(state).chunk(chunks=2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std +  1)
        std = torch.exp(log_std)
        return mean, std

    def evaluate(self, state):
        mean, std = self(state)

        dist = SquashedNormal(mean, std, stable=True, threshold=10)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, dist

    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float16, device=self.device).unsqueeze(0)
        else:
            # Tensor
            state = state.unsqueeze(dim=0).to(self.device)
        mean, std = self(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            dist = SquashedNormal(mean, std, stable=True, threshold=10)
            action = dist.sample()

        return action


class Critic(Container):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=nn.ReLU(inplace=True), device=None,
                 use_popart=False, beta=1e-4):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation)

        if use_popart:
            self.use_popart = use_popart
            # add policy activation for non-linearity
            self.policy_activation = activation

            self.popart_layer1 = POPARTLayer(1, 1, beta)
            self.popart_layer2 = POPARTLayer(1, 1, beta)

        self.to(device)

    def forward(self, state, action, normalize=False):
        q_value_1 = self.soft_q_net_1(state, action)
        q_value_2 = self.soft_q_net_2(state, action)

        if normalize and self.use_popart:
            return self.popart_layer1(self.policy_activation(q_value_1)), self.popart_layer2(self.policy_activation(q_value_2))

        return q_value_1, q_value_2

    def get_normalized_targets(self, Y):
        ''' Return normalized target for 2 critics using corresponding scale and shift '''
        return self.popart_layer1.get_normalized_target(Y), self.popart_layer2.get_normalized_target(Y)

    def update_popart_parameters(self, Y):
        self.popart_layer1.update_parameters(Y)
        self.popart_layer2.update_parameters(Y)


Actor = PolicyNetwork


class POPARTLayer(Container):
    def __init__(self, input_features, output_features, beta) -> None:
        super().__init__()
        self.beta = beta

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))

        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))

        nu = self.mu**2 + self.sigma**2
        self.register_buffer('nu', nu.requires_grad_(False))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        normalized_x = x.mm(self.weight.t())
        normalized_x += self.bias.unsqueeze(0).expand_as(normalized_x)

        return normalized_x

    @torch.no_grad()
    def get_normalized_target(self, Y):
        return (Y - self.mu) / self.sigma

    def update_parameters(self, Y):
        oldmu = self.mu
        oldsigma = self.sigma

        self.mu = (1 - self.beta) * oldmu + self.beta * Y.mean()
        self.nu = (1 - self.beta) * self.nu + self.beta * (Y**2).mean()
        self.nu = torch.clamp(self.nu, min=1e-4)
        self.sigma = torch.sqrt(self.nu - self.mu**2)

        self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma
