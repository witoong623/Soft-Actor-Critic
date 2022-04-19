# I got some helps from
# https://github.com/pytorch/vision/blob/0963ff715001a2bd27d235fd0f80df38a48eacf1/torchvision/models/resnet.py
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networkbase import NetworkBase


class BottleneckBlock(nn.Module):
    # expansion of output
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        
        self.diff_size = stride != 1 or in_channel != out_channel * self.expansion
        # when connection between difference input and output size
        # use linear projection, in this case it is 1x1 convolution
        if self.diff_size:
            self.ws = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion),
            )
        
        # building block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False),
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        # F(X)
        out = self.conv(X)
        if self.diff_size:
            X = self.ws(X)

        # F + x
        y = out + X
        return self.relu(y)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        
        self.diff_size = stride != 1 or in_channel != out_channel
        # when connection between difference input and output size
        # use linear projection, in this case it is 1x1 convolution
        if self.diff_size:
            self.ws = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        
        # building block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU()
        
    def forward(self, X):
        # F(X)
        out = self.conv(X)
        if self.diff_size:
            X = self.ws(X)

        # F + x
        y = out + X
        return self.relu(y)

class ResNet(nn.Module):
    def __init__(self, block, block_num, n_classes=10):
        super().__init__()
        
        self.input_channel = 64
        self.head = nn.Sequential(
            nn.Conv2d(3, self.input_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.conv2 = self._make_conv_layers(block, output_channel=64, time=block_num[0])
        self.conv3 = self._make_conv_layers(block, output_channel=128, time=block_num[1])
        self.conv4 = self._make_conv_layers(block, output_channel=256, time=block_num[2])
        self.conv5 = self._make_conv_layers(block, output_channel=512, time=block_num[3])
        # kernel size 4 so that output is 1x1
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(self.input_channel, n_classes)
        
    def _make_conv_layers(self, block, output_channel, time):
        layers = []
        stride = 1
        
        # downsample, use stride 2
        if self.input_channel != output_channel:
            stride = 2

        layers.append(block(self.input_channel, output_channel, stride=stride))
        
        self.input_channel = output_channel * block.expansion
        for _ in range(1, time):
            # reminding layer, stride 1 because no downsample afterward
            layers.append(block(self.input_channel, output_channel))
            
        return nn.Sequential(*layers)
        
        
    def forward(self, X):
        x = self.head(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        # reshape for FC linear (num_sample, features)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class BackboneResNet(NetworkBase):
    def __init__(self, input_channels, block, block_num):
        super().__init__()

        self.layers = []
        self.input_channel = 64
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_channels, self.input_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        ))
        self.layers.append(self._make_conv_layers(block, output_channel=64, time=block_num[0]))
        self.layers.append(self._make_conv_layers(block, output_channel=128, time=block_num[1]))
        self.layers.append(self._make_conv_layers(block, output_channel=256, time=block_num[2]))
        self.layers.append(self._make_conv_layers(block, output_channel=512, time=block_num[3]))
        
        self.features = nn.Sequential(*self.layers)

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, 1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def _make_conv_layers(self, block, output_channel, time):
        layers = []
        stride = 1
        
        # downsample, use stride 2
        if self.input_channel != output_channel:
            stride = 2

        layers.append(block(self.input_channel, output_channel, stride=stride))

        self.input_channel = output_channel * block.expansion
        for _ in range(1, time):
            # reminding layer, stride 1 because no downsample afterward
            layers.append(block(self.input_channel, output_channel))

        return nn.Sequential(*layers)

    def forward(self, X):
        x = self.features(X)
        x = self.output_layer(x)
        x = torch.flatten(x, start_dim=1)

        return x


def build_resnet18(n_classes=10):
    return ResNet(BasicBlock, block_num=[2, 2, 2, 2], n_classes=n_classes)

def build_resnet34(n_classes=10):
    return ResNet(BasicBlock, block_num=[3, 4, 6, 3], n_classes=n_classes)

def build_resnet50(n_classes=10):
    return ResNet(BottleneckBlock, block_num=[3, 4, 6, 3], n_classes=n_classes)

def build_resnet101(n_classes=10):
    return ResNet(BottleneckBlock, block_num=[3, 4, 23, 3], n_classes=n_classes)

def build_resnet152(n_classes=10):
    return ResNet(BottleneckBlock, block_num=[3, 8, 36, 3], n_classes=n_classes)

def backbone_resnet18(input_channels):
    return BackboneResNet(input_channels, BasicBlock, block_num=[2, 2, 2, 2])

def backbone_resnet34():
    return BackboneResNet(BasicBlock, block_num=[3, 4, 6, 3])


if __name__ == '__main__':
    model = backbone_resnet18(3)
    dummy = torch.randn((1, 3, 256, 512))

    output = model(dummy)
    print(output.size())
