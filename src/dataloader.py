import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PerformanceDataset(Dataset):
    def __init__(self, root_dir, target_length=10800, is_test=False):
        self.file_paths = self.get_file_paths(root_dir)
        self.target_length = target_length
        self.is_test = is_test

    def get_file_paths(self, root_dir):
        file_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('performance_monitor.csv'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def clean_data(self, data):
        # Convert non-numeric values to NaN, then forward and backward fill
        data = pd.to_numeric(data, errors='coerce')
        data = data.ffill().bfill()
        
        # If there are still NaNs, replace them with 0
        data = data.fillna(0)
        return data

    def interpolate_data(self, data):
        # Select only CPU and GPU utilization from the Value column based on the Kind column
        cpu_data = data[data['Kind'] == 0]['Value']
        gpu_data = data[data['Kind'] == 2]['Value']

        # Ensure no NaN after selection and clean data
        cpu_data = self.clean_data(cpu_data)
        gpu_data = self.clean_data(gpu_data)

        # If either CPU or GPU data is empty, fill with zeros
        if len(cpu_data) == 0:
            cpu_data = pd.Series([0] * self.target_length)
        if len(gpu_data) == 0:
            gpu_data = pd.Series([0] * self.target_length)

        # Stack CPU and GPU data into a single dataframe
        data = pd.DataFrame({'CPU': cpu_data, 'GPU': gpu_data})

        # Handle cases where the data length exceeds or falls short of target length
        current_length = len(data)
        if current_length > self.target_length:
            start_idx = (current_length - self.target_length) // 2
            data = data.iloc[start_idx : start_idx + self.target_length]
        else:
            # Interpolate to target length
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, self.target_length)
            interpolated_data = np.zeros((self.target_length, 2))

            for i, column in enumerate(['CPU', 'GPU']):
                interpolated_data[:, i] = np.interp(x_new, x_old, data[column])

            data = pd.DataFrame(interpolated_data, columns=['CPU', 'GPU'])

        # Final NaN cleaning (if any)
        data = data.fillna(0)
        return data

    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val + 1e-8)

    def augment_data(self, data):
        # Add Gaussian noise for augmentation
        noise = np.random.normal(0, 0.01, data.shape)
        data_noisy = data + noise
        data_noisy = np.clip(data_noisy, 0, 1)
        return data_noisy

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = pd.read_csv(file_path)

        # Apply interpolation and cleaning
        data = self.interpolate_data(data)
        data = self.normalize_data(data)

        if not self.is_test:
            data = self.augment_data(data)

        data_tensor = torch.tensor(data.values, dtype=torch.float32)
        return data_tensor

def create_dataloader(root_dir, batch_size=32, is_test=False):
    dataset = PerformanceDataset(root_dir=root_dir, is_test=is_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)
    return dataloader


