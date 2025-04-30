import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    """Convolutional actor-critic network for PPO."""
    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        self.actor  = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.actor(x)
        value  = self.critic(x)
        return logits, value

    def get_action(self, state: torch.Tensor):
        """Sample an action, its log-prob, entropy and value estimate."""
        logits, value = self(state)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
