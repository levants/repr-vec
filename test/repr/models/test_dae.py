"""
Created on Sep 17, 2019

Test case for auto-encoder call

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import torch

from repr.models.dae import DAEModel
from repr.models.input_utils import init_transforms
from repr.models.resnet import resnet50


class TestDAE(unittest.TestCase):
    """Test de-noising auto-encoder model"""

    def setUp(self) -> None:
        encoder = resnet50(pretrained=True)
        self.model = DAEModel(encoder)
        self.peprocessors = init_transforms(224, 224)
        img = np.zeros([100, 100, 3], dtype=np.uint8)
        img.fill(255)
        self.img = img

    def test_fast_forward(self):
        """Test fast-forward call of model"""
        input_tens = self.peprocessors(self.img)
        input_batch = torch.unsqueeze(input_tens, 0)
        output_tens = self.model(input_batch)

        print(output_tens.size())
