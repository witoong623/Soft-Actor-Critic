import torch
import torch.nn as nn


class CarlaConvVAE(nn.Module):
    def __init__(self, image_size, input_channels, n_hidden_channels,
                 latent_size, activation=nn.ReLU(inplace=True), device=None):
        super().__init__()

        self.activation = activation if activation is not None else nn.ReLU(inplace=True)

        self.last_encoder_output_channels = n_hidden_channels[-1]

        encoder_layers = []
        first_out_channel = 32
        encoder_layers.extend([
            nn.Conv2d(input_channels, first_out_channel, 3, stride=2, bias=False),
            nn.BatchNorm2d(first_out_channel),
            self.activation
        ])

        n_hidden_channels = [first_out_channel, *n_hidden_channels]

        def conv_block(n_in, n_out):
            return [
                nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(n_out),
                self.activation,
                nn.Conv2d(n_out, n_out, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(n_out),
                self.activation,
            ]

        for i in range(len(n_hidden_channels) - 1):
            conv_layer = conv_block(n_hidden_channels[i],
                                    n_hidden_channels[i + 1])

            encoder_layers.extend(conv_layer)

        self.encoder = nn.Sequential(*encoder_layers)

        (self.encoded_h, self.encoded_w), size_hist = self._calculate_spatial_size(image_size, self.encoder)

        self.mu = nn.Linear(self.last_encoder_output_channels * self.encoded_h * self.encoded_w, latent_size)
        self.logvar = nn.Linear(self.last_encoder_output_channels * self.encoded_h * self.encoded_w, latent_size)

        self.latent = nn.Linear(latent_size, self.last_encoder_output_channels * self.encoded_h * self.encoded_w)

        # TODO: generate this from encoder_layers
        self.settings = [
            # out channel, k, s, p
            (first_out_channel, 3, 2, 0),
            (64, 3, 2, 0),
            (64, 3, 2, 0),
            (128, 3, 2, 0),
            (128, 3, 2, 0),
            (256, 3, 2, 0),
            (256, 3, 2, 0),
        ]

        decoder_settings = self._generate_transposed_conv_settings(size_hist)

        next_block_output_channels = input_channels
        decoders = []
        for i, (in_channel, k, s, p, out_p) in enumerate(decoder_settings, 1):
            # order is reverse here because when calling reverse later, layer inside block will order correctly
            decoders.extend([
                self.activation,
                nn.BatchNorm2d(next_block_output_channels),
                nn.ConvTranspose2d(in_channel, next_block_output_channels, k, stride=s, padding=p, output_padding=out_p),
            ])

            # activation of last (it's first in reverse order) layer must be sigmoid
            if i == 1:
                decoders[0] = nn.Tanh()

            next_block_output_channels = in_channel

        self.decoder = nn.Sequential(*reversed(decoders))

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        z = self.latent(z)
        z = z.view(-1, self.last_encoder_output_channels, self.encoded_h, self.encoded_w)
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

    def _calculate_spatial_size(self, image_size, conv_layers):
        ''' Calculate spatial size after convolution layers '''
        H, W = image_size
        size_hist = []
        size_hist.append((H, W))

        for layer in conv_layers:
            if layer.__class__.__name__ != 'Conv2d':
                continue
            conv = layer
            H = int((H + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)
            W = int((W + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1)

            size_hist.append((H, W))

        return (H, W), size_hist

    def _generate_transposed_conv_settings(self, size_hist):
        ''' Generate transposed convolution setting in the same order as convolution setting '''
        decoder_settings = []
        # don't use the last size hist because it is the size after last conv
        # at each transposed conv, we use desired size to calculate output padding
        for (output_channels, kernel, stride, padding), (desired_H, desired_W) in zip(self.settings, size_hist[:-1]):
            output_padding_h = (desired_H + 2 * padding - kernel) % 2
            output_padding_w = (desired_W + 2 * padding - kernel) % 2
            output_padding = (output_padding_h, output_padding_w)

            decoder_settings.append((output_channels, kernel, stride, padding, output_padding))

        return decoder_settings


if __name__ == '__main__':
    model = CarlaConvVAE((256, 512), 6, [64, 128, 256], latent_size=768, activation=nn.ELU())

    dummy_input = torch.rand((1, 6, 256, 512))

    outs = model(dummy_input)

    print([item.size() for item in outs])
