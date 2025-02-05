import numpy as np
import torch
import matplotlib.pyplot as plt
from data_process import PedestrianDataset
from model import PedestrianSpeedNN

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
data_folder = "data/Corridor_Data"
dataset = PedestrianDataset(data_folder)

# load model
input_size = 1 + 10 * 2
model = PedestrianSpeedNN(input_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# test mean spacing in (0.5m ~ 3.5m)
num_samples = 100
mean_spacing_test = np.linspace(50, 350, num_samples)

k_neighbors = 10
test_features = []

for mean_spacing in mean_spacing_test:
    feature_vector = np.zeros(21)
    feature_vector[0] = mean_spacing  # mean spacing

    # generate 10 neighbors on radius = mean spacing
    angles = np.linspace(0, 2 * np.pi, k_neighbors, endpoint=False)  # uniform distribution on a circle
    radii = np.random.normal(mean_spacing, mean_spacing * 0.01, k_neighbors)  # disturb a bit
    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)

    # calculate x, y in feature_vector
    feature_vector[1:k_neighbors+1] = dx
    feature_vector[k_neighbors+1:] = dy

    # normalize
    norm_features = (feature_vector - dataset.feature_min) / (dataset.feature_max - dataset.feature_min + 1e-6)
    test_features.append(norm_features)

# to Tensor
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

# predict speed
with torch.no_grad():
    pred_speed_norm = 0.5*model(test_features_tensor).numpy()  # normalized

# denormalize speed
pred_speed = pred_speed_norm * (dataset.speed_max - dataset.speed_min) + dataset.speed_min

# plot
plt.figure(figsize=(8, 6))
plt.plot(0.01*mean_spacing_test, 0.01*pred_speed, label="Predicted Speed", color='b')
plt.xlabel("Mean Spacing [m]")
plt.ylabel("Predicted Speed [m/s]")
plt.title("Mean Spacing vs. Predicted Speed (ANN)")
plt.legend()
plt.grid()
plt.show()