import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PerformanceDataset(Dataset):
    def __init__(self, root_dir, target_length=28800):
        self.file_paths = self.get_file_paths(root_dir)
        self.target_length = target_length

    def get_file_paths(self, root_dir):
        file_paths = []
        # 遍历四层文件夹
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('performance_monitor.csv'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def interpolate_data(self, data, target_length):
        # 仅提取 CPU 占有率（index=0）和 GPU 占有率（index=2）
        data = data.iloc[:, [0, 2]]
        data = data.interpolate(method='linear', limit_direction='both')
        
        # 插值到相同长度
        current_length = len(data)
        if current_length < target_length:
            interpolated_data = data.reindex(
                pd.Series(range(target_length)),
                method='nearest',
                fill_value='extrapolate'
            )
        else:
            step = current_length / target_length
            interpolated_data = data.iloc[(step * pd.Series(range(target_length))).astype(int)]

        return interpolated_data.to_numpy()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = pd.read_csv(file_path)
        data = self.interpolate_data(data, self.target_length)
        return data

# 创建数据加载器
root_dir = 'data'
dataset = PerformanceDataset(root_dir=root_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

