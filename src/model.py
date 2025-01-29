import torch
import torch.nn as nn

class PedestrianSpeedNN(nn.Module):
    def __init__(self, input_size, hidden_size=3):
        super(PedestrianSpeedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # output speed
        )

    def forward(self, x):
        return self.model(x)
