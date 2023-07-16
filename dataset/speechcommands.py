#!/usr/bin/env python3

import os
import torch
import torchaudio

# import Dataset
from torch.utils.data import Dataset

# import speech commans dataset
import torchaudio.datasets.SPEECHCOMMANDS as speechcommands


class SpeechCommands(Dataset):
    def __init__(self, root, url="speech_commands_v0.02", folder_in_archive="SpeechCommands", download=False):
        super().__init__()
        self.dataset = speechcommands(root, url, folder_in_archive, download=download)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)




if __name__ == "__main__":
    sc = SpeechCommands(root="./data", download=True)
    print(f'Number of data points: {len(sc)}')
    print(f'Sample rate of the data: {sc[0][1]}')
    print(f'Labels of the data: {sc[0][2]}')
    print(f'Size of the waveform: {sc[0][0].size()}')
    print(f'First 10 samples: {sc[0][0][0:10]}')

