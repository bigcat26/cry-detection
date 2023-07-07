import torch
import librosa
import numpy as np

from torch import nn

class Log2Transform(nn.Module):
    def __init__(self):
        super(Log2Transform, self).__init__()

    def forward(self, x):
        eps = 1e-9
        x = torch.log2(x + eps)
        return x

class MonoTransform(nn.Module):
    def __init__(self):
        super(MonoTransform, self).__init__()

    def forward(self, x):
        # convert stereo to mono
        if x.size()[0] > 1:
            return torch.mean(x, dim=0).unsqueeze(0)
        return x

class NormalizeTransform(nn.Module):
    def __init__(self):
        super(NormalizeTransform, self).__init__()

    def forward(self, x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min)

class TrimTransform(nn.Module):
    def __init__(self, top_db=30, frame_length=1024, hop_length=512):
        super(TrimTransform, self).__init__()
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        
    def forward(self, x):
        trim, index = librosa.effects.trim(
            x.detach().numpy(), 
            top_db=self.top_db, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
            )
        return torch.Tensor(trim)

class NoiseGeneratorTransform(nn.Module):
    def __init__(self, noise_std=1e-6, gamma=10, milestones=[10, 30, 60]):
        super(NoiseGeneratorTransform, self).__init__()
        self.last_epoch = 0
        self.gamma = gamma
        self.noise_std = noise_std
        self.milestones = milestones

    def forward(self, x):
        noise = torch.randn(x.size()) * self.noise_std
        return x + noise

    def get_level(self):
        return self.noise_std
    
    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.noise_std *= self.gamma

class UnsqueezeTransform(nn.Module):
    def __init__(self):
        super(UnsqueezeTransform, self).__init__()

    def forward(self, x):
        return np.expand_dims(x, 0)

class FixedValueTransform(nn.Module):
    def __init__(self, value):
        super(FixedValueTransform, self).__init__()
        self.value = value

    def forward(self, x):
        return self.value

class BinaryTransform(nn.Module):
    def __init__(self, true_value_id):
        super(BinaryTransform, self).__init__()
        self.true_value_id = true_value_id

    def forward(self, x):
        return 1 if x == self.true_value_id else 0
