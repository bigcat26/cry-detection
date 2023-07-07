import numpy as np
import pandas as pd
import torch
import pickle as pkl 

from torch.utils.data import Dataset
from typing import Optional
from collections.abc import Callable

class CryDataset(Dataset):
    def __init__(self, 
                 pickle_file, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.transform = transform
        self.target_transform = target_transform

        with open(pickle_file, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx]['value'])
        label = self.data[idx]['target']

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data, label