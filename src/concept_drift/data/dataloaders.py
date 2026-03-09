from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size: int, shuffle: bool):
    return DataLoader(ArrayDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)
