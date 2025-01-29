import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PedestrianDataset(Dataset):
    def __init__(self, data_folder, k_neighbors=10):
        self.data = []
        self.k_neighbors = k_neighbors
        self._load_data(data_folder)

    def _load_data(self, folder):
        all_data = []
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
               lines = [list(map(float, line.strip().split())) for line in f]

            # 按帧号分组
            frame_dict = {}
            for row in lines:
                frame = int(row[1])
                if frame not in frame_dict:
                    frame_dict[frame] = []
                frame_dict[frame].append(row)

            # 处理每一帧数据
            for frame, pedestrians in frame_dict.items():
                if len(pedestrians) < self.k_neighbors:  # 如果人数不足 10，跳过
                   continue
            
                pedestrians = np.array(pedestrians)
                positions = pedestrians[:, 2:4]  # 仅提取 X, Y 坐标

                # 计算速度
                speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)

                # 取 K 近邻并存储
                for i in range(len(positions)):
                    distances = np.linalg.norm(positions - positions[i], axis=1)
                    nearest_indices = distances.argsort()[1:self.k_neighbors+1]  # 选取最近 10 个邻居
                
                    if len(nearest_indices) < self.k_neighbors:  # 如果邻居不足 10 个，跳过
                        continue
                
                    mean_spacing = distances[nearest_indices].mean()
                    input_features = np.concatenate(([mean_spacing], positions[nearest_indices].flatten()))  # 仅 X, Y

                    if len(input_features) != 21:  # 确保维度正确
                        print(f"跳过异常样本，实际维度 {len(input_features)}，预期 21")
                        continue

                    output_speed = speeds[i] if i < len(speeds) else speeds[-1]
                    all_data.append((input_features, output_speed))

        # 转换为 Tensor
        self.data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in all_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(data_folder, batch_size=32, shuffle=True, test_size=0.2):
    dataset = PedestrianDataset(data_folder)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

