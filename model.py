import torch.nn as nn
from config import *


expansion = 4


class Bottleneck(nn.Module):

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.base_width = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 4)
        self.layer2 = self.make_layer(128, 15, stride=2)
        self.layer3 = self.make_layer(256, 15, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * expansion, NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, out_ch, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != out_ch * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_ch * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * expansion),
            )

        layers = [Bottleneck(self.inplanes, out_ch, stride=stride, downsample=downsample)]
        self.inplanes = out_ch * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
