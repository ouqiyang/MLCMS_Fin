import torch
import torch.optim as optim
import torch.nn as nn
from model import PedestrianSpeedNN,WiedmannNNModel
from data_process import get_dataloader,get_wiedmann_dataloader

ann_losses = []
wiedmann_losses = []

# train params
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_loader, test_loader = get_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)
train_loader_wiedmann, _ = get_wiedmann_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)


# model init
input_size = 1 + 10 * 2  # 1 mean_spacing + 10 neighbor (x, y)
model = PedestrianSpeedNN(input_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


wiedmann_model = WiedmannNNModel().to(DEVICE)
optimizer_wiedmann = optim.Adam(wiedmann_model.parameters(), lr=LEARNING_RATE)

loss_fn = nn.MSELoss()

# train epoch
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    ann_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

# saving model
torch.save(model.state_dict(), "model.pth")


### **Train WiedmannNNModel**
for epoch in range(EPOCHS):
    wiedmann_model.train()
    total_loss = 0
    for inputs, targets in train_loader_wiedmann:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer_wiedmann.zero_grad()
        outputs = wiedmann_model(inputs).squeeze()
        loss = loss_fn(outputs, targets)

        # Check if loss is NaN
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            continue

        loss.backward()

        # Check for NaN in gradients
        for name, param in wiedmann_model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients for {name}")

        optimizer_wiedmann.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader_wiedmann)
    wiedmann_losses.append(avg_loss)
    print(f"WiedmannNNModel Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader_wiedmann)}")

torch.save({'ann_losses': ann_losses, 'wiedmann_losses': wiedmann_losses}, "loss_data.pth")