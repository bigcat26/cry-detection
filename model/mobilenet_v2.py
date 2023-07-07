import torch
import torch.nn as nn


def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expand_ratio)

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers_list = []
        if expand_ratio != 1:
            layers_list.append(conv_bn_act(in_channels, hidden_dim, kernel_size=1))

        layers_list.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000, width_multiplier=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        self.input_channels, self.input_height, self.input_width = input_shape

        # 设置超参数
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            # [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        if inverted_residual_setting is None:
            inverted_residual_setting = self.cfgs

        # 第一个卷积层
        self.features = [conv_bn_act(self.input_channels, int(32 * width_multiplier), kernel_size=3, stride=2, padding=1)]

        # 中间的Inverted Residual块
        input_channel = int(32 * width_multiplier)
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # 最后的卷积层和全局平均池化层
        self.features.append(conv_bn_act(input_channel, int(1280 * width_multiplier), kernel_size=1))
        self.features = nn.Sequential(*self.features)

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(1280 * width_multiplier), num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
