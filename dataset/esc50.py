import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url
from tqdm import tqdm
import pandas as pd
import torchaudio
import os
import torch.nn as nn


class ESC50(Dataset):
    base_folder = 'ESC-50-master'
    url = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
    filename = "ESC-50-master.zip"
    zip_md5 = '70cce0ef1196d802ae62ce40db11b620'
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': 'meta/esc50.csv',
        'md5': '54a0d0055a10bb7df84ad340a148722e',
    }

    def __init__(self, root, train: bool=True, download: bool=True, transform: nn.Module = None, target_transform: nn.Module = None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self._load_meta(train, download)

        self.data = []
        self.targets = []
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            wav, sr = torchaudio.load(file_path)
            wav = wav if not transform else torch.Tensor(transform(wav).data)
            self.data.append(wav)

            target = self.class_to_idx[row[self.label_col]]
            target = target if not target_transform else target_transform(target).data
            self.targets.append(target)

    def _load_meta(self, train: bool, download: bool, validate_fold: int = 5):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not os.path.exists(path) and download:
            self.download()

        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')

        data = pd.read_csv(path)
        index = data['fold'] != validate_fold if train else data['fold'] == validate_fold
        self.df = data[index]
        self.class_to_idx = {}
        self.classes = sorted(self.df[self.label_col].unique())
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        mel_spec, target = self.data[index], self.targets[index]
        return mel_spec, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            return False
        path = os.path.join(self.root, self.base_folder, self.audio_dir)
        if len(next(os.walk(path))[2]) != self.num_files_in_dir:
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.zip_md5)

        # extract file
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)
