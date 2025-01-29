from data_process import get_dataloader

# 加载数据
dataloader = get_dataloader("data/Corridor_Data", batch_size=4)

# 获取一个 batch 数据
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# 打印检查
print("输入数据形状: ", inputs.shape)  # 应该是 (batch_size, 1 + 10*2)  -->  (4, 21)
print("目标速度形状: ", targets.shape)  # 应该是 (batch_size,)  -->  (4,)
print("示例输入数据: ", inputs[0])  # 检查格式是否正确
print("示例目标速度: ", targets[0])
