import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        encoder = [
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*16*16
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*8*8
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 16*4*4
        ]

        bottleneck = [
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32)
        ]

        decoder = [
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*8*8
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*16*16
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
            # 1*32*32
        ]

        self.encoder = nn.Sequential(*encoder)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out