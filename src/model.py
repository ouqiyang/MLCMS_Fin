import torch
import torch.nn as nn
import numpy as np


class PedestrianSpeedNN(nn.Module):
    """Neural Network Model for Pedestrian Speed Prediction"""
    def __init__(self, input_size, hidden_size=8, min_speed=0, max_speed=2):  
        super(PedestrianSpeedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.min_speed = min_speed
        self.max_speed = max_speed

    def forward(self, x):
        speed = self.model(x)

        # De-normalize speed to original scale
        speed = speed * (self.max_speed - self.min_speed) + self.min_speed  

        return speed


class WiedmannNNModel(nn.Module):
    def __init__(self, min_speed=0, max_speed=2):  # Adjust range if needed
        super(WiedmannNNModel, self).__init__()
        self.hidden_layer = nn.Linear(1, 8)  
        self.v0_layer = nn.Linear(8, 1)
        self.l_layer = nn.Linear(8, 1)
        self.T_layer = nn.Linear(8, 1)

        # Store min/max speed for de-normalization
        self.min_speed = min_speed
        self.max_speed = max_speed

    def forward(self, mean_spacing):
        x = torch.relu(self.hidden_layer(mean_spacing))  
        v0 = torch.nn.functional.softplus(self.v0_layer(x))  
        l = torch.nn.functional.softplus(self.l_layer(x))
        T = torch.nn.functional.softplus(self.T_layer(x))

        v0 = torch.clamp(v0, min=0.1)
        T = torch.clamp(T, min=0.1)
        l = torch.clamp(l, min=0.1)
        mean_spacing = torch.clamp(mean_spacing, min=0.1)

        exponent = (l - mean_spacing) / (v0 * T)
        exponent = torch.clamp(exponent, min=-10, max=10)

        speed = v0 * (1 - torch.exp(exponent))

        # De-normalize speed
        speed = speed * (self.max_speed - self.min_speed) + self.min_speed  

        return speed

class WiedmannFixedModel:
    """Fixed Wiedmann Fundamental Diagram Model with Predefined Parameters"""
    def __init__(self, v0=1.3, T=0.5, l=0.3):
        self.v0 = v0
        self.T = T
        self.l = l

    def predict(self, mean_spacing):
        """Computes pedestrian speed using fixed Wiedmann parameters."""
        return self.v0 * (1 - np.exp((self.l - mean_spacing) / (self.v0 * self.T)))