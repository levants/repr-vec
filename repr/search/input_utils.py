"""
Created on Jun 27, 2018

Standalone module for features search

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from fastai.vision import *
from torchvision import transforms


class Resize(object):
    """Image processor before model"""

    def __init__(self, h=224, w=224, interpolation=cv2.INTER_AREA):
        super(Resize, self).__init__()
        self.interpolation = interpolation
        self.h = h
        self.w = w

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize input
            Args:
                img: input image
            Returns:
                re-sized image
        """
        return cv2.resize(img, (self.w, self.h), interpolation=self.interpolation)

    def __call__(self, *args, **kwargs):
        return self.resize(*args, **kwargs)


class CenterCrop(object):
    """Image processor before model"""

    def __init__(self, h=224, w=224, percnt: float = 0.1, interpolation=cv2.INTER_AREA):
        super(CenterCrop, self).__init__()
        self.h = h
        self.w = w
        self.percnt = percnt
        self.interpolation = interpolation

    @staticmethod
    def _resize_to(img, h: int, w: int, use_min: bool = False):
        "Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e w*h)"
        oh, ow = img.shape[:2]
        min_sz = (min if use_min else max)(oh, ow)
        hr = h / min_sz
        wr = w / min_sz
        th, tw = int(h * hr), int(w * wr)

        return th, tw

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Crop input
            Args:
                img: input image
            Returns:
                img_cnt: center cropped image
        """
        oh, ow = img.shape[:2]
        rh = int(oh * self.percnt)
        rw = int(ow * self.percnt)
        img_cnt = img[rh:oh - rh, rw:ow - rw]
        img_cnt = cv2.resize(img_cnt, (self.w, self.h), interpolation=self.interpolation)

        return img_cnt

    def __call__(self, *args, **kwargs):
        return self.crop(*args, **kwargs)


class Scale(object):

    def __init__(self):
        super(Scale, self).__init__()

    def scale(self, img: np.ndarray) -> np.ndarray:
        """
        Scales image
        Args:
            img: input image

        Returns:
            scaled image
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def __call__(self, *args, **kwargs):
        return self.scale(*args, **kwargs)


def init_transforms(h: int = 224, w: int = 224, percnt: float = 0.1,
                    interpolation: int = cv2.INTER_AREA, crop_center: bool = False) -> transforms:
    """
    Initializes transformations for network model inputs
    Args:
        h: input height
        w: input width
        percnt: percent of cropping borders
        interpolation: interpolation for image resizing
        crop_center: flag to crop images to center

    Returns:
        input tensor converter for fast froward call
    """
    return transforms.Compose([Scale(),
                               CenterCrop(h, w, percnt=percnt, interpolation=interpolation) if crop_center else \
                                   Resize(h, w, interpolation=interpolation),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
