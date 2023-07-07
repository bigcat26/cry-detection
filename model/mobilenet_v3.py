import torch
import torch.nn as nn


def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn_act(in_channels, out_channels):
    return conv_bn_act(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, squeeze_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(squeeze_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers_list = []
        if expand_ratio != 1:
            layers_list.append(conv_1x1_bn_act(in_channels, hidden_dim))
        layers_list.extend([
            conv_bn_act(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, int(in_channels / 4)),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileNetV3(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000, width_multiplier=1.0, dropout_rate=0.2):
        super(MobileNetV3, self).__init__()
        self.input_channels, self.input_height, self.input_width = input_shape

        # 设置超参数
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            # [3, 24, 3, 2],
            # [3, 40, 3, 2],
            [6, 40, 4, 1],
            [6, 80, 2, 2],
            # [6, 80, 3, 1],
            # [6, 80, 3, 1],
            # [6, 112, 3, 1],
            # [6, 112, 4, 1],
            # [6, 160, 1, 2],
            # [6, 160, 2, 1],
            # [6, 160, 2, 1]
        ]

        # 计算网络的最终输出通道数
        last_channels = int(1280 * width_multiplier) if width_multiplier > 1.0 else 1280

        # 第一个卷积层
        self.features = [conv_bn_act(self.input_channels, 16, kernel_size=3, stride=2, padding=1)]

        # 中间的Inverted Residual块
        input_channel = 16
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # 最后的卷积层和全局平均池化层
        self.features.append(conv_1x1_bn_act(input_channel, last_channels))
        self.features = nn.Sequential(*self.features)

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
