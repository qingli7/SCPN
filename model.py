import torch
import torchvision
import torch.nn as nn
from ResNet_Cifar import resnet18_cbam


class resnet18_gcpl(nn.Module):
    def __init__(self, num_classes, in_channel, latent_dim=2):
        super(resnet18_gcpl, self).__init__()
        cls = torchvision.models.resnet18(pretrained=True)
        # cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
        self.cls = nn.Sequential(*list(cls.children())[:-1])
        self.fc = nn.Linear(512, latent_dim, bias=False)
        
        self.prototypes = nn.Parameter(
            num_classes * torch.randn((num_classes, latent_dim)) - num_classes / 2., requires_grad=True)
        print(self.prototypes.dtype)
        
    def forward(self, x):
        # self.cls(x): [64, 512, 1, 1]
        # feat: [64, 512]
        feat = self.cls(x).reshape(x.shape[0], -1)
        feat = self.fc(feat)
        
        return feat


class resnet18_cifar(nn.Module):
    def __init__(self, num_classes, in_channel, latent_dim=2):
        super(resnet18_cifar, self).__init__()
        self.feature = resnet18_cbam()
        if in_channel == 1:
            self.feature.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
        # self.fc = nn.Linear(512, latent_dim, bias=False)
        
        self.prototypes = nn.Parameter(
            num_classes * torch.randn((num_classes, latent_dim)) - num_classes / 2., requires_grad=True)
        
        A = torch.randn(latent_dim, 512)
        self.A = nn.Parameter(A, requires_grad=True)

    def forward(self, x):
        # self.cls(x): [64, 512, 1, 1]
        # feat: [64, 512]
        feat = self.feature(x)
        # feat = self.fc(feat)
        return feat



if __name__ == '__main__':
    in_channel = 3
    num_classes = 10
    model = resnet18_gcpl(num_classes, in_channel, latent_dim=2)
    # model = resnet18_cbam(num_classes, in_channel, latent_dim=2)
    
    model.eval()

    x = torch.rand((64, 3, 32, 32))
    feat = model(x)
    print('feat.shape', feat.shape, 'model.prototypes.shape', model.prototypes.shape)




