import torch
from model import PedestrianSpeedNN
from data_process import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
input_size = 1 + 10 * 2
model = PedestrianSpeedNN(input_size).to(DEVICE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 加载测试数据
test_loader = get_dataloader("data\Corridor_Data", batch_size=32, shuffle=False)[1]

# 计算测试误差
total_loss = 0
loss_fn = torch.nn.MSELoss()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

print(f"Test Loss: {total_loss / len(test_loader)}")
