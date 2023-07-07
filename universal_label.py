
# labels = ['others', 'dog', 'cat', 'crying', 'car_horn', 'drilling']

class ESC50LabelTransform(nn.Module):
    def __init__(self, mapping=[0] * 50):
        super(ESC50LabelTransform, self).__init__()
        self.mapping = mapping
        self.mapping[0] = 1
        self.mapping[5] = 2
        self.mapping[20] = 3
        self.mapping[43] = 4

    def forward(self, x):
        return self.mapping[x]

class US8KLabelTransform(nn.Module):
    def __init__(self, mapping=[0] * 10):
        super(US8KLabelTransform, self).__init__()
        self.mapping = mapping
        self.mapping[1] = 4
        self.mapping[3] = 1
        self.mapping[4] = 5

    def forward(self, x):
        return self.mapping[x]
