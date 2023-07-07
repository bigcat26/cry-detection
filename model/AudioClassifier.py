import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, h=200, w=200, classes=50, num_conv_layers=2):
        super(AudioClassifier, self).__init__()
        
        self.features = nn.Sequential()
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = 32 * (2 ** i)
            self.features.add_module(f'conv{i+1}', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.features.add_module(f'bn{i+1}', nn.BatchNorm2d(out_channels))
            self.features.add_module(f'relu{i+1}', nn.ReLU(inplace=True))
            self.features.add_module(f'pool{i+1}', nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.fc1 = nn.Linear(h//2**num_conv_layers * w//2**num_conv_layers * out_channels, 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
