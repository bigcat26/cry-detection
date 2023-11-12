from torch.nn import Conv2d, Linear
from torchvision.models import mobilenet_v2

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
