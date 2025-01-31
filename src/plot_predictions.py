import torch
import matplotlib.pyplot as plt

# Load loss data
loss_data = torch.load("loss_data.pth")
ann_losses = loss_data['ann_losses']
wiedmann_losses = loss_data['wiedmann_losses']

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ann_losses) + 1), ann_losses, label="PedestrianSpeedNN (ANN)", marker="o")
plt.plot(range(1, len(wiedmann_losses) + 1), wiedmann_losses, label="WiedmannNNModel", marker="s")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()
plt.show()
