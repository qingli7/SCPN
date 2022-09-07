import torch
import torch.nn as nn
import torch.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MyModel(nn.Module):
    def __init__(self, in_channel, feat_dim, num_classes):
        super(MyModel, self).__init__()
        self.feature = nn.Sequential(*[
            ConvBlock(in_channel=in_channel, out_channel=16),
            ConvBlock(in_channel=16, out_channel=32),
            ConvBlock(in_channel=32, out_channel=64),
            ConvBlock(in_channel=64, out_channel=128),
            ConvBlock(in_channel=128, out_channel=256),
            ConvBlock(in_channel=256, out_channel=512)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feat_dim)

        self.prototypes = nn.Parameter(
            num_classes * torch.randn((num_classes, feat_dim)) - num_classes / 2., requires_grad=True)

    def forward(self, x):
        x = self.feature(x)                # shape: batch, 512, 4, 4
        x = self.pool(x)                   # shape: batch, 512, 1, 1
        x = x.reshape(x.shape[0], -1)      # shape: batch, 512
        x = self.fc(x)                     # shape: batch, latent_dim
        return x, self.prototypes


if __name__ == '__main__':
    model = MyModel(in_channel=1, feat_dim=2, num_classes=10).eval()  # in_channel 1 or 3， feat_dim 原型特征维度
    x = torch.rand(10, 1, 28, 28)
    out, prototype = model(x)
    print('out shape', out.shape)
    print('prototype shape: ', prototype, prototype.shape)     ###