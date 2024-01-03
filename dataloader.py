import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class MonDataLoaderAllData(Dataset):
    def __init__(self, data, lookback_size, lookforward_size, scaler=None):
        
        self.lookback_size = lookback_size
        self.lookforward_size = lookforward_size
        self.data = data
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(data)
        else:
            self.scaler = None

        d = []
        for i in range(0, 1 + len(data) - (self.lookback_size+ self.lookforward_size), 1):
            seq_x = self.data[i:i+self.lookback_size, :]
            seq_y = self.data[i+self.lookback_size:i+self.lookback_size+self.lookforward_size, :]
            d.append((seq_x, seq_y))
        self.data = d

    def __len__(self):
        return len(self.data) - self.lookback_size - self.lookforward_size + 1

    def __getitem__(self, idx):
      seq_x, seq_y = self.data[idx]
      return torch.tensor(seq_x), torch.tensor(seq_y)
      