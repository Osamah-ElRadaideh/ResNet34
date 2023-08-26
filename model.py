import torch.nn as nn
import numpy as np
import torch
'''implentation of the ResNet34 network https://arxiv.org/pdf/1512.03385.pdf'''
class Conv_block(nn.Module):
    #inital conv 7x7 block
    def __init__(self, in_channels, out_channels,kernel_size=7, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x 
    



class ResBlock(nn.Module):
    #the general residual block
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        res = x 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        x += res
        return self.relu(x)
    


class ResNet(nn.Module):
    '''the Residual network, layers represents the number each resblock is repeated, resnet 34 is comprised of 3,4,6,3 repitions
    '''
    def __init__(self,block= ResBlock, layers=(3,4,6,3), n_classes=4):
        super().__init__()
        self.in_channels = 64
        self.conv_block = Conv_block(3, self.in_channels)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.ff = nn.Linear(512, n_classes)
        
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            self.in_channels = out_channels
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0,-1,1,2)
        x = self.conv_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.ff(x)
        