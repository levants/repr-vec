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
