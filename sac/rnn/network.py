import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from common.network_base import VanillaLSTMNetwork


DEVICE_CPU = torch.device('cpu')


class ValueNetwork(VanillaLSTMNetwork):
    def __init__(self, state_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                 activation=F.relu, device=DEVICE_CPU):
        super().__init__(n_dims_before_lstm=[state_dim, *hidden_dims_before_lstm],
                         n_dims_lstm_hidden=hidden_dims_lstm,
                         n_dims_after_lstm=[*hidden_dims_after_lstm, 1],
                         skip_connection=skip_connection,
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim

    def forward(self, state, hidden=None):
        return super().forward(state, hidden)


class SoftQNetwork(VanillaLSTMNetwork):
    def __init__(self, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                 activation=F.relu, device=DEVICE_CPU):
        super().__init__(n_dims_before_lstm=[state_dim + action_dim, *hidden_dims_before_lstm],
                         n_dims_lstm_hidden=hidden_dims_lstm,
                         n_dims_after_lstm=[*hidden_dims_after_lstm, 1],
                         skip_connection=skip_connection,
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action, hidden=None):
        return super().forward(torch.cat([state, action], dim=-1), hidden)


class PolicyNetwork(VanillaLSTMNetwork):
    def __init__(self, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                 activation=F.relu, device=DEVICE_CPU, log_std_min=-20, log_std_max=2):
        super().__init__(n_dims_before_lstm=[state_dim, *hidden_dims_before_lstm],
                         n_dims_lstm_hidden=hidden_dims_lstm,
                         n_dims_after_lstm=[*hidden_dims_after_lstm, 2 * action_dim],
                         skip_connection=skip_connection,
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state, hidden=None):
        out, hidden = super().forward(state, hidden)
        mean, log_std = out.chunk(chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        return mean, std, hidden

    def evaluate(self, state, hidden=None, epsilon=1E-6):
        mean, std, hidden = self(state, hidden)

        z = Normal(0, 1).sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1.0 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, hidden

    def get_action(self, state, hidden=None, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
            mean, std, hidden = self(state, hidden)

            if deterministic:
                action = torch.tanh(mean)
            else:
                z = Normal(0, 1).sample()
                action = torch.tanh(mean + std * z)
        action = action.cpu().numpy()[0, 0]
        return action, hidden

    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=self.action_dim)
