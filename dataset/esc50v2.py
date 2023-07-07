import numpy as np
import pandas as pd
import torch
import torchaudio
import pickle as pkl 

from torch.utils.data import Dataset
from typing import Optional
from collections.abc import Callable

class ESC50Dataset(Dataset):
    def __init__(self, 
                 pickle_file, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.transform = transform
        self.target_transform = target_transform

        with open(pickle_file, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data) * 9
    
    def __getitem__(self, idx):
        rec_offset = idx // 9
        data_offset = idx % 9
        data = torch.Tensor(self.data[rec_offset]['value'][data_offset])
        # label = torch.Tensor([self.data[rec_offset]['target']])
        label = self.data[rec_offset]['target']
        
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data, label