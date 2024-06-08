import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, data_dir, bc_points_file, bc_labels_file, ic_points_file, ic_labels_file):
        self.bc_points = np.load(os.path.join(data_dir, bc_points_file))
        self.bc_labels = np.load(os.path.join(data_dir, bc_labels_file))
        self.ic_points = np.load(os.path.join(data_dir, ic_points_file))
        self.ic_labels = np.load(os.path.join(data_dir, ic_labels_file))

        self.bc_len = len(self.bc_points)
        self.ic_len = len(self.ic_points)

    def __len__(self):
        return max(self.bc_len, self.ic_len)

    def __getitem__(self, idx):
        bc_idx = idx % self.bc_len
        ic_idx = idx % self.ic_len

        # 合并边界条件和初始条件
        points = np.concatenate((self.bc_points[bc_idx], self.ic_points[ic_idx]), axis=0)
        labels = np.concatenate((self.bc_labels[bc_idx], self.ic_labels[ic_idx]), axis=0)

        # 确保点的数量是 3
        points = points[:3]
        labels = labels[:3]

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def create_datasets(data_dir):
    train_dataset = FlowDataset(
        data_dir,
        bc_points_file="bc_points.npy",
        bc_labels_file="bc_label.npy",
        ic_points_file="ic_points.npy",
        ic_labels_file="ic_label.npy"
    )
    return train_dataset

def create_test_dataset(data_dir):
    inputs = np.load(os.path.join(data_dir, 'eval_points.npy'))
    labels = np.load(os.path.join(data_dir, 'eval_label.npy'))
    return inputs, labels
