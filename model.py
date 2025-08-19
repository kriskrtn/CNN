import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

n_classes = 10

class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()
        self.cv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.cv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)

        self.ap = nn.AvgPool2d(kernel_size=8)

        self.flat = nn.Flatten()
        self.linear = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.bn2(self.cv2(out))

        residual = self.cv3(x)
        out += residual

        out = self.activation(out)

        out = self.flat(self.ap(out))
        out = self.linear(out)


        return out