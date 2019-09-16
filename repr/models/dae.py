"""
Created on Jul 12, 2019

Denoising autoencoder for representation learning

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from fastai.vision import *
from fastai.vision.models.unet import PixelShuffle_ICNR

unet_learner()

class DAEModel(nn.Module):
    """De-noising auto-encoder for representation learning"""

    def __init__(self, encoder: nn.Module, out_features: int = 2048):
        super(DAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(get_conv(8*nf, 4*nf, ks, use_bn=True),
                                     PixelShuffle_ICNR(4*nf, 4*nf, scale=2, leaky=True),
                                     get_conv(4*nf, 2*nf, ks, use_bn=True),
                                     PixelShuffle_ICNR(2*nf, 2*nf, scale=2, leaky=True),
                                     get_conv(2*nf, nf, ks, use_bn=True),
                                     PixelShuffle_ICNR(nf, nf, scale=2, leaky=True),
                                     get_conv(nf, 3, ks, use_bn=False))
