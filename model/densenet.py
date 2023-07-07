import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(DenseNet, self).__init__()
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output
