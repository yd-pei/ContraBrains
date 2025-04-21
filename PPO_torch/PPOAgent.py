import torch.nn as nn

class PPOAgent(nn.Module):
    def __init__(self, action_dim, in_channels=4):
        super(PPOAgent, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # x shape = (batch, 4, 84, 84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)    # (batch, 64*7*7)
        x = self.fc(x)
        return self.actor(x), self.critic(x)