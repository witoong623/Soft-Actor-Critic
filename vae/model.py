import torch
import torch.nn as nn
import torch.nn.functional as F

from common.network import NetworkBase


class ConvVAE(NetworkBase):
    def __init__(self, image_size,  input_channel=3, latent_size=64):
        super(ConvVAE, self).__init__()

        self.beta = 3

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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, output_padding=1),
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


if __name__ == '__main__':
    image_size = (96, 96)
    model = ConvVAE(image_size, latent_size=128)
    dummy = torch.zeros((1, 3, *image_size))

    reconstruct, mu, logvar = model(dummy)
    print('reconstruct', reconstruct.size())
