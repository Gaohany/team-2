

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_classes: int=10, drop_rate: float=0.0):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ac1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ac2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * self.conv2.out_channels, 120)
        self.fc2 = nn.Linear(self.fc1.out_features, 84)
        self.fc3 = nn.Linear(self.fc2.out_features, num_classes)
        self.deactivate_activation = True
    def forward(self, x):
        batch_size = x.shape[0]
        ' \n        x = self.conv_1(x)\n        self.conv_1 = nn.Sequential(\n            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), \n            nn.ReLU(), \n            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))\n        '
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.avgpool1(x)
        ' \n        if self.dropout > 0:\n            x = F.dropout(x, p=self.dropout)\n        '
        x = self.dropout(x)
        ' \n        x = self.conv_2(x)\n        self.conv_2 = nn.Sequential(\n            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), \n            nn.ReLU(), \n            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))\n        '
        x = self.conv2(x)
        x = self.ac2(x)
        x = self.avgpool2(x)
        ' \n        if self.dropout > 0:\n            x = F.dropout(x, p=self.dropout)\n        '
        x = self.dropout(x)
        ' \n        x = x.view(batch_size, 4 * 4 * 16)\n        '
        x = x.view(batch_size, 4 * 4 * 16)
        ' \n        x = self.fc_1(x)\n        self.fc_1 = nn.Linear(4 * 4 * 16, 120)\n        '
        
        x = self.fc1(x)
        ' \n        if self.dropout > 0:\n            x = F.dropout(x, p=self.dropout)\n        '
        x = self.dropout(x)
        ' \n        x = self.fc_2(self.relu(x))\n        self.fc_2 = nn.Linear(120, 84)\n        '
        if not self.deactivate_activation:
            x = self.relu(x)
        x = self.fc2(x)
        ' \n        if self.dropout > 0:\n            x = F.dropout(x, p=self.dropout)\n        '
        x = self.dropout(x)
        ' \n        feature_vector = self.fc_3(self.relu(x))\n        self.fc_3 = nn.Linear(84, num_classes)\n        '
        x = self.relu(x)
        feature_vector = self.fc3(x)
        ' \n        x = F.log_softmax(feature_vector, dim=1)\n        '
        x = F.log_softmax(feature_vector, dim=1)
        return x

def count_parameters(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
if __name__ == '__main__':
    m = Net()
    print(count_parameters(m))