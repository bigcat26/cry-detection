#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    import google.colab
    IN_COLAB = True
    drive.mount('/content/drive')
except:
    IN_COLAB = False


# In[2]:


import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


# In[3]:


from torch.nn import Conv2d, Linear
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, alexnet, SqueezeNet

def get_mobilenet_v2(in_channels=1, num_classes=50, **kwargs):
    '''
    create a mobilenet_v2 model with specified number of input channels and output classes
    '''
    model = mobilenet_v2(**kwargs)
    #model.features[0][0] = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.features[0][0] = Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #model.classifier[1] = Linear(in_features=1280, out_features=1000, bias=True)
    model.classifier[1] = Linear(in_features=1280, out_features=num_classes, bias=True)
    return model

def get_mobilenet_v3_small(in_channels=1, num_classes=50, **kwargs):
    '''
    create a mobilenet_v3 model with specified number of input channels and output classes
    '''
    model = mobilenet_v3_small(**kwargs)
    model.features[0][0] = Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[3] = Linear(in_features=1024, out_features=num_classes, bias=True)
    return model

def get_mobilenet_v3_large(in_channels=1, num_classes=50, **kwargs):
    '''
    create a mobilenet_v3 model with specified number of input channels and output classes
    '''
    model = mobilenet_v3_large(**kwargs)
    model.features[0][0] = Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[3] = Linear(in_features=1280, out_features=num_classes, bias=True)
    return model

def get_alexnet(in_channels=1, num_classes=50, **kwargs):
    model = alexnet(**kwargs)
    model.features[0] = Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
    model.classifier[6] = Linear(in_features=4096, out_features=num_classes, bias=True)
    return model

def get_squeezenet(in_channels=1, num_classes=50, **kwargs):
    model = SqueezeNet(**kwargs)
    model.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
    model.classifier[1] = Conv2d(512, 50, kernel_size=(1, 1), stride=(1, 1))
    return model


# In[6]:


import torch
from torch import nn

class Log2Transform(nn.Module):
    def __init__(self):
        super(Log2Transform, self).__init__()

    def forward(self, x):
        eps = 1e-9
        x = torch.log2(x + eps)
        return x

import torch
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, noise_std=0.1):
        self.original_dataset = original_dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        x, y = self.original_dataset[index]

        noise_level = self.noise_std * (index / len(self.original_dataset))

        noisy_x = x + torch.randn_like(x) * noise_level

        return noisy_x, y


# In[7]:


# @title
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        log_dir: str = None,
        save_dir: str = "./checkpoints",
        save_epochs: int = 0,
        regularization: str = None,
        regularization_alpha: float = 0.01,
        amp: bool = False,
        early_stop: Callable = None
        ):

        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.regularization = regularization
        self.regularization_alpha = regularization_alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device=self.device)

        # auto mixed precision
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()

        if log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_epochs = save_epochs
        self.early_stop = early_stop


    def _compute_regularization(self, alpha=0.01):
        if self.regularization == 'L1':
            return alpha * sum(torch.norm(param, 1) for param in self.model.parameters())
        elif self.regularization == 'L2':
            return alpha * sum(torch.norm(param, 2) for param in self.model.parameters())
        elif self.regularization == 'L2sqrt':
            return alpha * sum(torch.norm(param, 2) ** 2 for param in self.model.parameters())
        else:
            return 0.0

    def evaluate(self, dataloader: DataLoader, epoch: int):
        self.model.eval()

        total_loss = 0.0
        targets = []
        predictions = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                # 计入总损失
                total_loss += loss.item()
                # 记录预测结果
                predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
                # 记录真实结果
                targets.extend(y.argmax(dim=1).cpu().numpy())

        # 计算准确率
        correct_predictions = sum(torch.Tensor(predictions) == torch.Tensor(targets))
        accuracy = correct_predictions.item() / len(targets)

        # 将验证损失和准确度写入 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Validation/Loss', total_loss / len(dataloader), epoch)
            self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)

        return total_loss / len(dataloader), accuracy

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()

        total_loss = 0.0
        targets = []
        predictions = []

        for x, y in tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if self.amp:
                with autocast():
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)

                loss += self._compute_regularization(self.regularization_alpha)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss += self._compute_regularization(self.regularization_alpha)
                loss.backward()
                self.optimizer.step()

            # 计入总损失
            total_loss += loss.item()
            # 记录预测结果
            predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
            # 记录真实结果
            targets.extend(y.argmax(dim=1).cpu().numpy())

        # 计算准确率
        correct_predictions = sum(torch.Tensor(predictions) == torch.Tensor(targets))
        accuracy = correct_predictions.item() / len(targets)

        # 将训练损失和准确度写入 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Train/Loss', total_loss / len(dataloader), epoch)
            self.writer.add_scalar('Train/Accuracy', accuracy, epoch)

        return total_loss / len(dataloader), accuracy

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss, train_accuracy = self.train_epoch(self.train_dataloader, epoch)

            if self.val_dataloader is not None:
                val_loss, val_accuracy = self.evaluate(self.val_dataloader, epoch)
                print(f"Epoch {epoch + 1}/{start_epoch + epochs} => "
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{start_epoch + epochs} => "
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            if self.writer is not None:
                self.writer.flush()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.save_epochs > 0 and (epoch + 1) % self.save_epochs == 0:
                self.save_checkpoint(epoch + 1)

            if self.early_stop is not None and self.early_stop(val_loss):
                print("Early stopping triggered.")
                break

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(checkpoint, f'{self.save_dir}/{epoch}.pth')

    def load_checkpoint(self, file):
        state = torch.load(file)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def save_torchscript(self, file, trace_input_shape=None):
        if trace_input_shape is not None:
            mod = torch.jit.trace(self.model, torch.randn(trace_input_shape).to(self.device))
        else:
            mod = torch.jit.script(self.model)
        mod.save(file)

    def count_parameters(self):
        '''
        returns the number of trainable parameters in the self.model.
        '''
        # return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# In[8]:


import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url
from tqdm import tqdm
import pandas as pd
import librosa
import os
import torch.nn as nn

class ESC50(Dataset):
    base_folder = 'ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
    url = "https://github.com/karolpiczak/ESC-50/archive/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6.zip"
    filename = "ESC-50-master.zip"
    zip_md5 = '629e8e9ebc1592bcceb06da6bec40275'
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': 'meta/esc50.csv',
        'md5': '54a0d0055a10bb7df84ad340a148722e',
    }

    def __init__(self, cache_dir: str, local_dir: str=None, train: bool=True, download: bool=False, skip_ssl_verify: bool=False, transform: nn.Module = None, target_transform: nn.Module = None):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.local_dir = local_dir or self.cache_dir
        self._load_meta(train, download, skip_ssl_verify)

        self.data = []
        self.targets = []
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.cache_dir, self.base_folder, self.audio_dir, row[self.file_col])
            wav, sr = librosa.load(file_path)
            wav = torch.from_numpy(wav).unsqueeze(0) if not transform else torch.Tensor(transform(torch.from_numpy(wav).unsqueeze(0)).data)
            self.data.append(wav)

            target = self.class_to_idx[row[self.label_col]]
            target = target if not target_transform else target_transform(target).data
            self.targets.append(target)

    def _load_meta(self, train: bool, download: bool, skip_ssl_verify: bool, validate_fold: int = 5, check_files: bool = False):
        data = self.get_meta()
        if data is None and download:
            self.download(skip_ssl_verify=skip_ssl_verify)
            self.unzip()
            data = self.get_meta()

        if data is None:
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')

        if check_files and not self.check_all_files():
            raise RuntimeError('Dataset audio files not found or corrupted.' +
                               ' You can use download=True to download it')

        # split train set and validation set
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
    
    def get_meta(self):
        path = os.path.join(self.cache_dir, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            print(f"get_meta check_integrity failed, path={path} md5={self.meta['md5']}")
            return None
        
        return pd.read_csv(path)

    def check_all_files(self) -> bool:
        path = os.path.join(self.cache_dir, self.base_folder, self.audio_dir)
        if len(next(os.walk(path))[2]) != self.num_files_in_dir:
            return False
        return True

    def unzip(self) -> bool:
        # extract file
        from zipfile import ZipFile
        zip_file = os.path.join(self.local_dir, self.filename)
        if not check_integrity(zip_file, self.zip_md5):
            raise RuntimeError(f'{zip_file} is corrupted..')

        with ZipFile(zip_file, 'r') as zip:
            zip.extractall(path=self.cache_dir)

    def download(self, skip_ssl_verify: bool=False):
        zip_file = os.path.join(self.local_dir, self.filename)
        if check_integrity(zip_file, self.zip_md5):
            print("download check_integrity failed")
            return

        if skip_ssl_verify:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

        download_url(self.url, self.local_dir, self.filename, self.zip_md5)


# In[9]:


# %load_ext tensorboard
# %tensorboard --logdir logs


# In[10]:


import torch
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio.transforms import MelSpectrogram
from torch.optim.lr_scheduler import MultiStepLR

# from dataset.esc50 import ESC50
# from model.mobilenet_v2 import get_mobilenet_v2
# from utils.transform import Log2Transform
from torchvision.transforms import Compose
from torchvision.ops.focal_loss import sigmoid_focal_loss

# from utils.trainer import Trainer

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop



nclasses = 50
orig_freq = 44100
target_freq = 16000
n_fft = 1024
window_size = 20
hop_size = 10

learning_rate = 1e-2
train_batch_size = 200
val_batch_size = 50

window_length = int(round(window_size * target_freq / 1000))
hop_length = int(round(hop_size * target_freq / 1000))

resample = Resample(orig_freq=orig_freq, new_freq=target_freq)
spectrogram = MelSpectrogram(
    n_fft=n_fft,
    win_length=window_length,
    hop_length=hop_length,
    n_mels=256,
    )

transform = torch.nn.Sequential(resample, spectrogram, Log2Transform())
target_transform = Compose([lambda x: torch.eye(50)[x]])

# if IN_COLAB:
#     cache_dir = '/content/cache'
#     local_dir = '/content/drive/MyDrive/datasets'
# else:
#     cache_dir = 'e:/dataset'
#     local_dir = 'e:/dataset'
cache_dir = '/root/autodl-tmp/cache'
local_dir = '/root/autodl-tmp/dataset'
    
train = DataLoader(
    AugmentedDataset(ESC50(
    # ESC50(
        cache_dir=cache_dir,
        local_dir=local_dir,
        download=True,
        train=True,
        transform=transform,
        target_transform=target_transform
        )
    ),
    # pin_memory=True,
    # num_workers=4,
    # prefetch_factor=3,
    batch_size=train_batch_size,
    shuffle=True)

test = DataLoader(
    AugmentedDataset(ESC50(
    # ESC50(
          cache_dir=cache_dir,
          local_dir=local_dir,
          download=True,
          train=False,
          transform=transform,
          target_transform=target_transform
         )
    ),
    # pin_memory=True,
    # prefetch_factor=2,
    # num_workers=2,
    batch_size=val_batch_size,
    shuffle=False)

model = get_mobilenet_v3_large(weights=None, in_channels=1, num_classes=50)
# model = get_alexnet(weights=None, in_channels=1, num_classes=50)
# model = get_squeezenet(in_channels=1, num_classes=50)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
# scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
early_stop = EarlyStopping(patience=30, delta=0.001)

trainer = Trainer(
    model,
    # loss_fn=sigmoid_focal_loss,
    loss_fn=CrossEntropyLoss(),
    train_dataloader=train,
    val_dataloader=test,
    optimizer=optimizer,
    # lr_scheduler=scheduler,
    log_dir='/root/tf-logs',
    save_dir='./checkpoints/mobilenet_v3_large',
    amp=False,
    save_epochs=25,
    early_stop=early_stop,
    )

# trainer.load_checkpoint('./drive/MyDrive/checkpoints/100.pth')
trainer.train(start_epoch=0, epochs=1000)
# In[ ]:

# del trainer
# del model
# del train
# del test
