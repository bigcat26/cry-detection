# import argparse
import torch
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio.transforms import MelSpectrogram
from torch.optim.lr_scheduler import MultiStepLR

from dataset.esc50 import ESC50
from model.mobilenet_v2 import get_mobilenet_v2
from utils.transform import Log2Transform
from torchvision.transforms import Compose
from torchvision.ops.focal_loss import sigmoid_focal_loss

from utils.trainer import Trainer

nclasses = 50
orig_freq = 44100
target_freq = 16000
n_fft = 1024
window_size = 20
hop_size = 10

learning_rate = 1e-2
train_batch_size = 100
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

train = DataLoader(
    ESC50(
        root='E:/dataset',
        train=True,
        transform=transform,
        target_transform=target_transform
        ),
    batch_size=train_batch_size,
    shuffle=True)

test = DataLoader(
    ESC50(root='E:/dataset',
          train=False,
          transform=transform,
          target_transform=target_transform
         ),
    batch_size=val_batch_size,
    shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_mobilenet_v2(in_channels=1, num_classes=50)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
# scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

trainer = Trainer(
    model,
    # loss_fn=sigmoid_focal_loss,
    loss_fn=CrossEntropyLoss(),
    train_dataloader=train,
    val_dataloader=test,
    optimizer=optimizer,
    # lr_scheduler=scheduler,
    log_dir='./logs',
    amp=True,
    save_epochs=25,
    )

trainer.train(epochs=50)
