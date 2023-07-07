import torch
from torch.utils.data import Dataset

class NoiseDataset(Dataset):
    def __init__(self, shape, num_samples, target_id, noise_std=0.1):
        self.shape = shape
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.target_id = target_id

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        noise = torch.randn(*self.shape) * self.noise_std
        return noise, self.target_id
