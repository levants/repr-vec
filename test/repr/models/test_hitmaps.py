"""
Created on Sep 17, 2019

Test prediction hit-maps

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from repr.models.hitmaps import grad_cam
from repr.models.input_utils import init_transforms
from repr.models.resnet import resnet50


class TestHitmaps(unittest.TestCase):
    """Test de-noising auto-encoder model"""

    def setUp(self) -> None:
        self.model = resnet50(pretrained=True)
        self.peprocessors = init_transforms(224, 224)
        img = np.zeros([100, 100, 3], dtype=np.uint8)
        img.fill(255)
        self.img = img

    def test_fast_forward(self):
        """Test fast-forward call of model"""
        mult = grad_cam(self.model, self.img, 0, self.peprocessors)
        print(mult.size())
