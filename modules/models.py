from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinyAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class TinySimCLR(nn.Module):
    def __init__(self, projection_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.encoder(x))
        return F.normalize(z, dim=1)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.25,
    top1_margin: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature
    similarity.fill_diagonal_(-1e9)
    labels = torch.arange(batch, device=z.device)
    labels = torch.cat([labels + batch, labels])
    loss = F.cross_entropy(similarity, labels)
    cosine = torch.matmul(z1, z2.T)
    positive = cosine.diag()
    negative_mask = ~torch.eye(batch, dtype=torch.bool, device=z1.device)
    negative_similarity = cosine[negative_mask].mean()
    positive_similarity = positive.mean()
    strongest_negative = cosine.masked_fill(~negative_mask, -1e9).max(dim=1).values
    top1 = (positive > strongest_negative + top1_margin).float().mean()
    return loss, positive_similarity, negative_similarity, top1
