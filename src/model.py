import torch
import torch.nn as nn

class PedestrianSpeedNN(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(PedestrianSpeedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # output speed
        )

    def forward(self, x):
        return self.model(x)
