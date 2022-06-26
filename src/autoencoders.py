import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Optional, Tuple

class Autoencoder(nn.Module):
    def __init__(self, image_dim: Tuple[int, int, int], latent_dim: int = 32):
        super().__init__()
        self.encoder =  nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.shape = self.encoder(torch.ones((1,) + image_dim)).shape[1:]
        size = int(np.prod(self.shape))
        self.fc1 = nn.Linear(size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x / 255.0
        z = self._encoder(x)
        x = self._decoder(z)
        return x, z

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).reshape(x.size(0), -1)
        z = self.fc1(x)
        return z

    def _decoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(x).reshape((x.size(0),) + self.shape)
        x = self.decoder(x)
        return x

    def get_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x / 255.0
        with torch.no_grad():
            z = self._encoder(x)
            x = self._decoder(z)
        x = x.permute((0, 2, 3, 1))
        x = (255 * torch.clip(x, 0, 1)).to(torch.uint8).squeeze(0)
        return x, z

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = self._encoder(x).reshape(x.size(0), -1)
        x_no_grad = x.detach()
        return x_no_grad

    def calc_loss(self, reconstruction: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(reconstruction, img)


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=4,
                stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(3840, latent_dim)
        self.fc_log_var = nn.Linear(3840, latent_dim)
        self.fc = nn.Linear(latent_dim, 3840)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3840, 256, (4, 5), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 128, (4, 5), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 64, (4, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 3, (2, 2), stride=2, padding=1),
        )


    def sample(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_log_var(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = z.detach().cpu()
        return z.squeeze().numpy()


    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample


    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x / 255.0
        x = self.encoder(x)
        mu, log_var = self.fc_mu(x), self.fc_log_var(x)

        z = self.reparameterize(mu, log_var)
        z = self.fc(z)
        z = z.view(-1, 3840, 1, 1)

        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var


    def calc_loss(self, recon, x) -> torch.Tensor:
        x_hat, mean, log_var = recon
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD


        #return F.mse_loss(reconstruction, img)