import torch
import torch.optim as optim
import torch.nn as nn
from model import PedestrianSpeedNN
from data_process import get_dataloader

# train params
EPOCHS = 160
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l1_lambda = 1e-5
# load data
train_loader, test_loader = get_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)

# model init
input_size = 1 + 10 * 2  # 1 mean_spacing + 10 neighbor (x, y)
model = PedestrianSpeedNN(input_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# train epoch
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        loss += l1_regularization(model, l1_lambda)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

# saving model
torch.save(model.state_dict(), "model.pth")
