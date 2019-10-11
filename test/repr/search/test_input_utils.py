"""
Created on Oct 11, 2019

Test case for image resizing

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from fastai.vision import *

from repr.search.input_utils import CenterCrop


class TestInputUtils(unittest.TestCase):
    """Test de-noising auto-encoder model"""

    def setUp(self) -> None:
        img = np.zeros([1024, 1500, 3], dtype=np.uint8)
        img.fill(255)
        # self.img2 = cv2.imread('181.jpg', cv2.IMREAD_ANYCOLOR)
        self.img = img
        self.h = 512
        self.w = 512

    def test_fast_forward(self):
        """Test fast-forward call of model"""
        cr = CenterCrop(h=self.h, w=self.w)
        img_cnt = cr(self.img)
        # img_cnt2 = cr(self.img2)
        nh, nw = img_cnt.shape[:2]
        assert nh == self.h and nw == self.w, 'Height and width does not match'
        print(img_cnt.shape)
        # cv2.imshow('Centered image', img_cnt2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
