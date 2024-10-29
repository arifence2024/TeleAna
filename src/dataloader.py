import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PerformanceDataset(Dataset):
    def __init__(self, root_dir, target_length=10800, is_test=False):
        self.file_paths = self.get_file_paths(root_dir, is_test)
        self.target_length = target_length

    def get_file_paths(self, root_dir, is_test):
        file_paths = []
        if is_test:
            # Traverse one person's folder for testing
            for date_folder in os.listdir(root_dir):
                date_path = os.path.join(root_dir, date_folder)
                if os.path.isdir(date_path):
                    for root, dirs, files in os.walk(date_path):
                        for file in files:
                            if file.endswith('performance_monitor.csv'):
                                file_paths.append(os.path.join(root, file))
        else:
            # Traverse all folders for training
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('performance_monitor.csv'):
                        file_paths.append(os.path.join(root, file))
        return file_paths

    def interpolate_data(self, data):
        data = data.iloc[:, [0, 2]].fillna(method='ffill').fillna(method='bfill')
        
        # Convert columns to numeric (in case of string values)
        data = data.apply(pd.to_numeric, errors='coerce')
        
        current_length = len(data)

        if current_length > self.target_length:
            start_idx = (current_length - self.target_length) // 2
            data = data.iloc[start_idx : start_idx + self.target_length]
        else:
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, self.target_length)
            interpolated_data = np.zeros((self.target_length, 2))
            for i in range(2):
                interpolated_data[:, i] = np.interp(x_new, x_old, data.iloc[:, i])
            data = pd.DataFrame(interpolated_data, columns=['CPU', 'GPU'])

        return data

    def normalize_data(self, data):
        # Convert to numeric again to ensure numerical data
        data = data.apply(pd.to_numeric, errors='coerce')

        # Min-Max normalization
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)

    def augment_data(self, data):
        noise = np.random.normal(0, 0.01, data.shape)
        data_noisy = data + noise

        smooth_data = (data_noisy + np.roll(data_noisy, 1, axis=0) + np.roll(data_noisy, -1, axis=0)) / 3
        smooth_data = np.clip(smooth_data, 0, 1)
        return smooth_data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = pd.read_csv(file_path)

        # Ensure columns are numeric before interpolation
        data = data.apply(pd.to_numeric, errors='coerce')

        # Process data to 9 hours (10,800 time steps)
        data = self.interpolate_data(data)
        # Normalize data
        data = self.normalize_data(data)
        # Apply data augmentation
        data = self.augment_data(data)

        return data

def create_dataloader(root_dir, batch_size=32, is_test=False):
    dataset = PerformanceDataset(root_dir=root_dir, is_test=is_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)

