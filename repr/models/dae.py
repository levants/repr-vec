"""
Created on Sep 17, 2019

De-noising auto-encoder for representation learning

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fastai.vision import *


class Unflatten(nn.Module):

    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.view(x.size(0), *self.sizes)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Sequential block
        conv_block = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        ]
        self.conv_block = nn.Sequential(*conv_block)
        # relu func to apply after residual summation on features
        self.leak_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = x + self.conv_block(x)
        out = self.leak_relu(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Sequential block
        conv_block = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        ]
        self.block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        up_block = [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                    # upsample by factor of 2
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                    # drop channel dim. by factor of 2 for each up-sample step
                    nn.BatchNorm2d(in_channels // 2),
                    nn.LeakyReLU(inplace=True)
                    ]
        self.up = nn.Sequential(*up_block)

    def forward(self, x):
        out = self.up(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels=512, up_depth=6, output_image_channels=3, use_residual=False):
        super().__init__()
        block = []

        channels = in_channels
        if use_residual:
            for i in range(1, up_depth):
                block.append(UpBlock(channels))
                channels = int(in_channels / (2 ** i))
                if channels < 1:
                    raise ValueError(
                        "Channel dimension is less then 1! Increase channel dimension in input for decoder network.")
                block.append(ResidualBlock(channels))
        else:
            for i in range(1, up_depth):
                block.append(UpBlock(channels))
                channels = int(in_channels / (2 ** i))
                if channels < 1:
                    raise ValueError(
                        "Channel dimension is less then 1! Increase channel dimension in input for decoder network.")
                block.append(DecoderBlock(channels))

        self.block = nn.Sequential(*block)
        self.last = nn.Conv2d(channels, output_image_channels, kernel_size=1)

    def forward(self, x):
        x = self.block(x)
        out = self.last(x)
        return out


def get_deconv(nf: int, of: int, ks: int, stride: int = 1, opad: int = 0, use_bn: bool = True):
    deconv = nn.ConvTranspose2d(in_channels=nf,
                                out_channels=of,
                                kernel_size=ks,
                                stride=stride,
                                padding=ks // 2,
                                output_padding=opad,
                                bias=False)
    bn = nn.BatchNorm2d(of)
    act = nn.LeakyReLU(0.2, inplace=True)

    return nn.Sequential(deconv, bn, act) if use_bn else nn.Sequential(deconv, act)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def tranp_conv_decoder(nf: int, ks: int, z_dim: int):
    deconv1 = get_deconv(8 * nf, 4 * nf, ks, stride=2, opad=1)
    deconv2 = get_deconv(4 * nf, 2 * nf, ks, stride=2, opad=1)
    deconv3 = get_deconv(2 * nf, nf, ks, stride=2, opad=1)
    deconv4 = get_deconv(nf, 3, ks, use_bn=False)

    return nn.Sequential(nn.Linear(z_dim, 512 * 8 * 8),
                         Unflatten(512, 8, 8),
                         deconv1,
                         deconv2,
                         deconv3,
                         deconv4)


class DAEModel(nn.Module):
    """De-noising auto-encoder for representation learning"""

    def __init__(self, encoder: nn.Module, nf: int = 64, ks: int = 5, z_dim: int = 2048):
        super(DAEModel, self).__init__()
        self.encoder = nn.Sequential(*list(encoder.children())[:-2] + [Flatten()])
        self.decoder = tranp_conv_decoder(nf, ks, z_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        z = self.encoder(input_tensor)
        out_tensor = self.decoder(z)

        return out_tensor
