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
        mean_spacing_values = []
        feature_values = []
        speed_values = []

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                lines = [list(map(float, line.strip().split())) for line in f]

            frame_dict = {}
            for row in lines:
                frame = int(row[1])
                if frame not in frame_dict:
                    frame_dict[frame] = []
                frame_dict[frame].append(row)

            for frame, pedestrians in frame_dict.items():
                if len(pedestrians) < self.k_neighbors:
                    continue

                pedestrians = np.array(pedestrians)
                positions = pedestrians[:, 2:4]  # Extract X, Y coordinates
                speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)

                for i in range(len(positions)):
                    distances = np.linalg.norm(positions - positions[i], axis=1)
                    nearest_indices = distances.argsort()[1:self.k_neighbors+1]

                    if len(nearest_indices) < self.k_neighbors:
                        continue

                    mean_spacing = distances[nearest_indices].mean()
                    feature_vector = np.concatenate(([mean_spacing], positions[nearest_indices].flatten()))

                    # Ensure input size is exactly 21
                    if len(feature_vector) != 21:
                        print(f"Warning: Incorrect feature vector size {len(feature_vector)}, expected 21")
                        continue

                    mean_spacing_values.append(mean_spacing)
                    feature_values.append(feature_vector)
                    speed_values.append(speeds[i] if i < len(speeds) else speeds[-1])

        # Convert to numpy arrays
        mean_spacing_values = np.array(mean_spacing_values)
        feature_values = np.array(feature_values)
        speed_values = np.array(speed_values)

        self.feature_min = feature_values.min(axis=0)
        self.feature_max = feature_values.max(axis=0)
        self.speed_min = speed_values.min()
        self.speed_max = speed_values.max()

        for i in range(len(mean_spacing_values)):
            norm_features = (feature_values[i] - self.feature_min) / (self.feature_max - self.feature_min + 1e-6)
            norm_speed = (speed_values[i] - self.speed_min) / (self.speed_max - self.speed_min)

            self.data.append((torch.tensor(norm_features, dtype=torch.float32), torch.tensor(norm_speed, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class WiedmannDataset(Dataset):
    def __init__(self, data_folder, k_neighbors=10):
        self.data = []
        self.k_neighbors = k_neighbors
        self._load_data(data_folder)

    def _load_data(self, folder):
        all_data = []
        mean_spacing_values = []
        speed_values = []

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                lines = [list(map(float, line.strip().split())) for line in f]

            frame_dict = {}
            for row in lines:
                frame = int(row[1])
                if frame not in frame_dict:
                    frame_dict[frame] = []
                frame_dict[frame].append(row)

            for frame, pedestrians in frame_dict.items():
                if len(pedestrians) < self.k_neighbors:
                    continue
                
                pedestrians = np.array(pedestrians)
                positions = pedestrians[:, 2:4]  # Extract X, Y coordinates
                speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)

                for i in range(len(positions)):
                    distances = np.linalg.norm(positions - positions[i], axis=1)
                    nearest_indices = distances.argsort()[1:self.k_neighbors+1]

                    if len(nearest_indices) < self.k_neighbors:
                        continue

                    mean_spacing = distances[nearest_indices].mean()

                    # Prevent mean_spacing from being too small
                    mean_spacing = max(mean_spacing, 0.1)

                    if i < len(speeds):
                        actual_speed = speeds[i]
                    else:
                        actual_speed = speeds[-1]

                    mean_spacing_values.append(mean_spacing)
                    speed_values.append(actual_speed)

        # Normalize values
        mean_spacing_values = np.array(mean_spacing_values)
        speed_values = np.array(speed_values)

        self.mean_spacing_min = mean_spacing_values.min()
        self.mean_spacing_max = mean_spacing_values.max()
        self.speed_min = speed_values.min()
        self.speed_max = speed_values.max()

        for i in range(len(mean_spacing_values)):
            norm_spacing = (mean_spacing_values[i] - self.mean_spacing_min) / (self.mean_spacing_max - self.mean_spacing_min)
            norm_speed = (speed_values[i] - self.speed_min) / (self.speed_max - self.speed_min)
            self.data.append((torch.tensor([norm_spacing], dtype=torch.float32), torch.tensor(norm_speed, dtype=torch.float32)))

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

def get_wiedmann_dataloader(data_folder, batch_size=32, shuffle=True, test_size=0.2):
    dataset = WiedmannDataset(data_folder)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader