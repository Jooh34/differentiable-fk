import torch
import numpy as np

class FkoptDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.random.rand(300000, 3).astype(np.float32) * 2 - 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]