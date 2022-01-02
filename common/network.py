import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet import efficientnet_b0


__all__ = [
    'build_encoder',
    'Container', 'NetworkBase',
    'VanillaNeuralNetwork', 'VanillaNN',
    'MultilayerPerceptron', 'MLP',
    'GRUHidden', 'cat_hidden',
    'RecurrentNeuralNetwork', 'RNN',
    'ConvolutionalNeuralNetwork', 'CNN'
]


def build_encoder(config):
    state_dim = (config.state_dim or config.observation_dim)
    state_encoder = nn.Identity()
    if config.FC_encoder:
        if config.state_dim is not None or len(config.encoder_hidden_dims) > 0:
            state_encoder = VanillaNeuralNetwork(n_dims=[config.observation_dim,
                                                         *config.encoder_hidden_dims,
                                                         state_dim],
                                                 activation=config.encoder_activation,
                                                 output_activation=None)
    elif config.RNN_encoder:
        state_encoder = RecurrentNeuralNetwork(n_dims_before_rnn=[config.observation_dim,
                                                                  *config.encoder_hidden_dims_before_rnn],
                                               n_dims_rnn_hidden=config.encoder_hidden_dims_rnn,
                                               n_dims_after_rnn=[*config.encoder_hidden_dims_after_rnn,
                                                                 state_dim],
                                               skip_connection=config.skip_connection,
                                               trainable_initial_hidden=config.trainable_hidden,
                                               activation=config.encoder_activation,
                                               output_activation=None)
    elif config.CNN_encoder and config.env == 'CarRacing-v0':
        state_encoder = CarRacingCNN(image_size=(config.image_size, config.image_size),
                                     input_channels=config.observation_dim,
                                     output_dim=state_dim,
                                     headless=True,
                                     n_hidden_channels=config.encoder_hidden_channels,
                                     activation=config.encoder_activation,
                                     output_activation=None,
                                     **config.build_from_keys(['kernel_sizes',
                                                               'strides',
                                                               'paddings',
                                                               'poolings',
                                                               'batch_normalization']))
    elif config.CNN_encoder:
        state_encoder = ConvolutionalNeuralNetwork(image_size=(config.image_size, config.image_size),
                                                   input_channels=config.observation_dim,
                                                   output_dim=state_dim,
                                                   n_hidden_channels=config.encoder_hidden_channels,
                                                   activation=config.encoder_activation,
                                                   output_activation=None,
                                                   **config.build_from_keys(['kernel_sizes',
                                                                             'strides',
                                                                             'paddings',
                                                                             'poolings',
                                                                             'batch_normalization']))

    elif config.VAE_encoder:
        state_encoder = ConvVAE((270, 480), latent_size=512)
        if not config.weight_path:
            raise ValueError('--weight-path must be provided if encoder is VAE.')
        state_encoder.load_model(config.weight_path)
        state_encoder.eval()
    elif config.EFFICIENTNET_encoder:
        state_encoder = efficientnet_b0(pretrained=True, progress=True, use_modified_ver=True, input_channels=6)

    config.state_encoder = state_encoder
    config.state_dim = state_dim

    return state_encoder


class Container(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = None

    def to(self, *args, **kwargs):
        device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            for module in self.children():
                if isinstance(module, Container):
                    module.to(device)
            self.device = device
        return super().to(*args, **kwargs)

    def save_model(self, path, key_filter=None):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if key_filter is not None and not key_filter(key):
                state_dict.pop(key)
            else:
                state_dict[key] = state_dict[key].cpu()

        torch.save(state_dict, path)
        return state_dict

    def load_model(self, path, strict=True):
        return self.load_state_dict(torch.load(path, map_location=self.device), strict=strict)


NetworkBase = Container


class VanillaNeuralNetwork(NetworkBase):
    def __init__(self, n_dims, activation=nn.ReLU(inplace=True), output_activation=None, device=None):
        super().__init__()

        self.activation = activation
        self.output_activation = output_activation

        self.linear_layers = nn.ModuleList()
        for i in range(len(n_dims) - 1):
            self.linear_layers.append(module=nn.Linear(in_features=n_dims[i],
                                                       out_features=n_dims[i + 1],
                                                       bias=True))

        self.in_features = n_dims[0]
        self.out_features = n_dims[-1]

        self.to(device)

    def forward(self, x):
        n_layers = len(self.linear_layers)
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < n_layers - 1:
                x = self.activation(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class GRUHidden(object):
    def __init__(self, hidden):
        self.hidden = hidden

    def __str__(self):
        return str(self.hidden)

    def __repr__(self):
        return repr(self.hidden)

    def __getitem__(self, item):
        new_hidden = []
        for h in self.hidden:
            new_hidden.append(h[item])
        return GRUHidden(hidden=new_hidden)

    def __getattr__(self, item):
        attr = getattr(torch.Tensor, item)

        if callable(attr):
            self_hidden = self.hidden

            def func(*args, **kwargs):
                new_hidden = []
                for h in self_hidden:
                    new_hidden.append(attr(h, *args, **kwargs))
                return GRUHidden(hidden=new_hidden)

            return func
        else:
            new_hidden = []
            for h in self.hidden:
                new_hidden.append(getattr(h, item))
            return GRUHidden(hidden=new_hidden)

    def float(self):
        new_hidden = []
        for h in self.hidden:
            new_hidden.append(torch.FloatTensor(h))
        return GRUHidden(hidden=new_hidden)

    @staticmethod
    def cat(hiddens, dim=0):
        hiddens = [hidden.hidden for hidden in hiddens]
        new_hidden = []
        for ith_layer_hiddens in zip(*hiddens):
            ith_layer_hiddens = torch.cat(ith_layer_hiddens, dim=dim)
            new_hidden.append(ith_layer_hiddens)
        return GRUHidden(hidden=new_hidden)


cat_hidden = GRUHidden.cat


class RecurrentNeuralNetwork(NetworkBase):
    def __init__(self, n_dims_before_rnn, n_dims_rnn_hidden, n_dims_after_rnn,
                 skip_connection=False, trainable_initial_hidden=False,
                 activation=nn.ReLU(inplace=True), output_activation=None, device=None):
        assert len(n_dims_rnn_hidden) > 0

        super().__init__()

        n_dims_rnn_hidden = [n_dims_before_rnn[-1], *n_dims_rnn_hidden]
        n_dims_after_rnn = [n_dims_rnn_hidden[-1], *n_dims_after_rnn]

        self.skip_connection = skip_connection
        if skip_connection:
            n_dims_after_rnn[0] += n_dims_before_rnn[-1]

        self.activation = activation
        self.output_activation = output_activation

        self.linear_layers_before_rnn = VanillaNeuralNetwork(n_dims=n_dims_before_rnn,
                                                             activation=activation,
                                                             output_activation=None)

        self.gru_layers = nn.ModuleList()
        for i in range(len(n_dims_rnn_hidden) - 1):
            self.gru_layers.append(module=nn.GRU(input_size=n_dims_rnn_hidden[i],
                                                 hidden_size=n_dims_rnn_hidden[i + 1],
                                                 num_layers=1, bias=True,
                                                 batch_first=False, bidirectional=False))

        if trainable_initial_hidden:
            self.init_hiddens = nn.ParameterList()
            for i in range(len(n_dims_rnn_hidden) - 1):
                bound = 1 / np.sqrt(n_dims_rnn_hidden[i])
                hidden = nn.Parameter(torch.Tensor(1, 1, n_dims_rnn_hidden[i + 1]), requires_grad=True)
                nn.init.uniform_(hidden, -bound, bound)
                self.init_hiddens.append(hidden)
        else:
            self.init_hiddens = []
            for i in range(len(n_dims_rnn_hidden) - 1):
                self.init_hiddens.append(torch.zeros(1, 1, n_dims_rnn_hidden[i + 1],
                                                     device=torch.device('cpu'),
                                                     requires_grad=False))

        self.linear_layers_after_rnn = VanillaNeuralNetwork(n_dims=n_dims_after_rnn,
                                                            activation=activation,
                                                            output_activation=output_activation)

        self.in_features = self.linear_layers_before_rnn.in_features
        self.out_features = self.linear_layers_after_rnn.out_features

        self.to(device)

    def forward(self, x, hx=None):
        if hx is None:
            hx = self.initial_hiddens(batch_size=x.size(1))
        assert isinstance(hx, GRUHidden)

        identity = x = self.linear_layers_before_rnn(x)

        ha = []
        hn = []
        for i, gru_layer in enumerate(self.gru_layers):
            hn.append(None)
            x, hn[i] = gru_layer(x, hx.hidden[i])
            ha.append(x)

        if self.skip_connection:
            x = torch.cat([x, identity], dim=-1)
        x = self.linear_layers_after_rnn(x)
        ha = GRUHidden(hidden=ha)
        hn = GRUHidden(hidden=hn)
        return x, hn, ha

    def initial_hiddens(self, batch_size=1):
        init_hidden = GRUHidden(hidden=list(self.init_hiddens))
        init_hidden = init_hidden.to(self.device)
        return init_hidden.repeat(1, batch_size, 1)


class ConvolutionalNeuralNetwork(NetworkBase):
    def __init__(self, image_size, input_channels, n_hidden_channels,
                 kernel_sizes, strides, paddings, poolings,
                 output_dim=None, headless=False, batch_normalization=False,
                 activation=nn.ReLU(inplace=True), output_activation=None, device=None):
        assert len(n_hidden_channels) == len(kernel_sizes)
        assert len(n_hidden_channels) == len(strides)
        assert len(n_hidden_channels) == len(paddings)
        assert len(n_hidden_channels) == len(poolings)

        assert bool(output_dim) != bool(headless)

        super().__init__()

        n_hidden_channels = [input_channels, *n_hidden_channels]

        self.activation = activation
        self.output_activation = output_activation

        self.conv_layers = nn.ModuleList()
        for i in range(len(n_hidden_channels) - 1):
            conv_layer = nn.Conv2d(n_hidden_channels[i],
                                   n_hidden_channels[i + 1],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   bias=True)

            self.conv_layers.append(module=conv_layer)

        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(1, len(n_hidden_channels)):
                self.batch_norm_layers.append(module=nn.BatchNorm2d(n_hidden_channels[i],
                                                                    affine=True))

        self.max_pooling_layers = nn.ModuleList(list(map(nn.MaxPool2d, poolings)))

        dummy = torch.zeros(1, input_channels, *image_size)
        with torch.no_grad():
            dummy = self(dummy)
        conv_output_dim = int(np.prod(dummy.size()))

        if output_dim is not None:
            assert not headless
            self.linear_layer = nn.Linear(in_features=conv_output_dim,
                                          out_features=output_dim,
                                          bias=True)
            self.out_features = output_dim
        else:
            assert headless
            self.out_features = conv_output_dim
        self.in_features = (input_channels, *image_size)

        self.to(device)

    def forward(self, x):
        input_size = x.size()
        x = x.view(-1, *input_size[-3:])

        for i, (conv_layer, max_pooling_layer) in enumerate(zip(self.conv_layers,
                                                                self.max_pooling_layers)):
            x = conv_layer(x)
            if self.batch_normalization:
                x = self.batch_norm_layers[i](x)
            x = self.activation(x)
            x = max_pooling_layer(x)

        x = x.view(*input_size[:-3], -1)
        if hasattr(self, 'linear_layer'):
            x = self.linear_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class CarRacingCNN(NetworkBase):
    def __init__(self, image_size, input_channels, n_hidden_channels,
                 kernel_sizes, strides, paddings, poolings,
                 output_dim=None, headless=False, batch_normalization=False,
                 activation=nn.ReLU(inplace=True), output_activation=None, device=None):
        assert len(n_hidden_channels) == len(kernel_sizes)
        assert len(n_hidden_channels) == len(strides)
        assert len(n_hidden_channels) == len(paddings)
        assert len(n_hidden_channels) == len(poolings)

        super().__init__()

        n_hidden_channels = [input_channels, *n_hidden_channels]

        self.activation = activation
        self.output_activation = output_activation

        self.conv_layers = nn.ModuleList()
        for i in range(len(n_hidden_channels) - 1):
            conv_layer = nn.Conv2d(n_hidden_channels[i],
                                   n_hidden_channels[i + 1],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   bias=True)

            self.conv_layers.append(module=conv_layer)

        # for PPO that work, it is not used
        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(1, len(n_hidden_channels)):
                self.batch_norm_layers.append(module=nn.BatchNorm2d(n_hidden_channels[i],
                                                                    affine=True))

        dummy = torch.zeros(1, input_channels, *image_size)
        with torch.no_grad():
            dummy = self(dummy)
        conv_output_dim = int(np.prod(dummy.size()))

        if not headless:
            self.linear_layer = nn.Linear(in_features=conv_output_dim,
                                          out_features=output_dim,
                                          bias=True)
            self.out_features = output_dim
        else:
            assert conv_output_dim == output_dim
            self.out_features = conv_output_dim
        self.in_features = (input_channels, *image_size)

        self.to(device)

    def forward(self, x):
        input_size = x.size()
        x = x.view(-1, *input_size[-3:])

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.batch_normalization:
                x = self.batch_norm_layers[i](x)
            x = self.activation(x)

        x = x.view(*input_size[:-3], -1)
        if hasattr(self, 'linear_layer'):
            x = self.linear_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class ConvVAE(NetworkBase):
    def __init__(self, image_size,  input_channel=3, latent_size=64, beta=3):
        super(ConvVAE, self).__init__()

        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder_h, self.encoder_w = self._calculate_spatial_size(image_size, self.encoder)

        self.mu = nn.Linear(64 * self.encoder_h * self.encoder_w, latent_size)
        self.var = nn.Linear(64 * self.encoder_h * self.encoder_w, latent_size)

        self.latent = nn.Linear(latent_size, 64 * self.encoder_h * self.encoder_w)

        # control output size by controlling output_padding
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def _calculate_spatial_size(self, image_size, conv_layers):
        H, W = image_size

        for layer in conv_layers:
            if layer.__class__.__name__ != 'Conv2d':
                continue
            conv = layer
            H = (H + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1
            W = (W + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1

        return (int(H), int(W))


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64 * self.encoder_h * self.encoder_w)
        return self.mu(x), self.var(x)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def decode(self, z):
        z = self.latent(z)
        z = z.view(-1, 64, self.encoder_h, self.encoder_w)
        z = self.decoder(z)
        return z


    def forward(self, x, encode=False, mean=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # divide total loss by batch size
        return (recon_loss + self.beta * kl_diverge) / x.shape[0]


MLP = MultilayerPerceptron = VanillaNN = VanillaNeuralNetwork
RNN = RecurrentNeuralNetwork
CNN = ConvolutionalNeuralNetwork


if __name__ == '__main__':
    image_size = (270, 480)
    model = ConvVAE(image_size, latent_size=512)
    print(model)
    dummy = torch.zeros((1, 3, *image_size))

    reconstruct, mu, logvar = model(dummy)
    print('reconstruct', reconstruct.size())
