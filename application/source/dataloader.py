import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_ = torch.tensor(self.X[idx], dtype=torch.float32)
        y_ = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x_, y_
