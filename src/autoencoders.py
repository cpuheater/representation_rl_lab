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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
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
