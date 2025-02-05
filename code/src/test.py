import torch
import numpy as np
from model import PedestrianSpeedNN,WiedmannNNModel, WiedmannFixedModel
from data_process import get_dataloader,get_wiedmann_dataloader


# Store predictions
actual_speeds = []
ann_predictions = []
wiedmann_predictions = []
fixed_predictions = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
input_size = 1 + 10 * 2
model = PedestrianSpeedNN(input_size).to(DEVICE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

wiedmann_model = WiedmannNNModel().to(DEVICE)
wiedmann_model.load_state_dict(torch.load("wiedmann_nn.pth"))
wiedmann_model.eval()

# Load fixed Wiedmann model
fixed_model = WiedmannFixedModel()

# load test data
test_loader = get_dataloader("data/Corridor_Data", batch_size=32, shuffle=False)[1]
_, test_loader_wiedmann = get_wiedmann_dataloader("data/Corridor_Data", batch_size=32, shuffle=False)


# calculate MSE
total_loss = 0
loss_fn = torch.nn.MSELoss()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        actual_speeds.extend(targets.cpu().numpy())
        ann_predictions.extend(outputs.cpu().numpy())

print(f"Test Loss: {total_loss / len(test_loader)}")


### **Evaluate Wiedmann NN Model**
total_loss_wiedmann = 0
with torch.no_grad():
    for inputs, targets in test_loader_wiedmann:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = wiedmann_model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        total_loss_wiedmann += loss.item()

        wiedmann_predictions.extend(outputs.cpu().numpy())

print(f"Wiedmann NN Test Loss: {total_loss_wiedmann / len(test_loader_wiedmann)}")

### **Compare Fixed Wiedmann Model**
total_loss_fixed = 0
with torch.no_grad():
    for inputs, targets in test_loader_wiedmann:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs_fixed = torch.tensor([fixed_model.predict(x.item()) for x in inputs], dtype=torch.float32).to(DEVICE)
        loss = loss_fn(outputs_fixed, targets)
        total_loss_fixed += loss.item()

        fixed_predictions.extend(outputs_fixed.cpu().numpy())     

print(f"Wiedmann Fixed Test Loss: {total_loss_fixed / len(test_loader_wiedmann)}")

np.savez("test_results.npz",
         actual_speeds=actual_speeds,
         ann_predictions=ann_predictions,
         wiedmann_predictions=wiedmann_predictions,
         fixed_predictions=fixed_predictions)

print("Test results saved as 'test_results.npz'")