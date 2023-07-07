import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
	def __init__(self, num_classes=10, pretrained=True):
		super(ResNet, self).__init__()
		self.model = models.resnet50(pretrained=pretrained)
		self.model.fc = nn.Linear(2048, num_classes)
		
	def forward(self, x):
		output = self.model(x)
		return output