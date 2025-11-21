# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        self.stages = nn.ModuleList()
        out_ch = in_channels // len(pool_sizes)
        for p in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(p),
                nn.Conv2d(in_channels, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(x)
        return torch.cat(priors, dim=1)

class PSPNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet50', pool_sizes=[1,2,3,6], pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            channels = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            channels = 2048
        else:
            raise ValueError('backbone not supported')
        # remove FC layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.ppm = PyramidPoolingModule(channels, pool_sizes)
        self.final = nn.Sequential(
            nn.Conv2d(channels + channels // len(pool_sizes) * len(pool_sizes), 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)  # upsample to input approx
        return x
