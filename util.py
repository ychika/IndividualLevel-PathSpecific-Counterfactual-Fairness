import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from torch.utils.data import Dataset, DataLoader

class GermanDataset(Dataset):
    """German dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 1:]
        out_label = self.data[idx, 0]
        return out_data, out_label
    
class AdultDataset(Dataset):
    """Adult dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 1:]
        out_label = self.data[idx, 0]
        return out_data, out_label

class SynthDataset(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 0:4] # A, M, D, Q
        out_label = self.data[idx, 4]
        return out_data, out_label

class SynthDataset_r(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, [1,3]] # M, Q
        out_label = self.data[idx, 4]
        return out_data, out_label

class SynthMDataset(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 0:3] # A, R, M
        out_label = self.data[idx, 3] # Y
        return out_data, out_label

class SynthMDataset_r(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 1:3] # R, M
        out_label = self.data[idx, 3] # M
        return out_data, out_label

class SynthUDataset(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, 0:6] # A, M, D, Q1, Q2, Q3
        out_label = self.data[idx, 6]
        return out_data, out_label

class SynthUDataset_r(Dataset):
    """Synthetic dataset."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx, [1,3,4]] ## only M, Q1, Q2
        out_label = self.data[idx, 6]
        return out_data, out_label