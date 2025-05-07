"""CNN feature extractor + actor & critic heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class PPOAgent(nn.Module):
    """Simple IMPALA‑style CNN followed by policy & value heads."""

    def __init__(self, obs_channels: int, num_actions: int):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self._linear_dim = 64 * 7 * 7
        self._policy = nn.Sequential(
            nn.Linear(self._linear_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )
        self._value = nn.Sequential(
            nn.Linear(self._linear_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self._encoder(x)
        x = x.view(x.size(0), -1)
        return self._policy(x), self._value(x)