"""
Created on Jun 27, 2018

Standalone module for features search

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
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


def init_transforms(h=224, w=224, interpolation=cv2.INTER_AREA) -> transforms:
    """
    Initializes transformations for network model inputs
    Args:
        h: input height
        w: input width
        interpolation: interpolation for image resizing

    Returns:
        input tensor converter for fast froward call
    """
    return transforms.Compose([Scale(),
                               Resize(h, w, interpolation=interpolation),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def prepare(preprocessors: transforms, *imgs: np.ndarray):
    """
    Generate batch of tensors from array
    Args:
        preprocessors: input tensor preprocessors
        *imgs: input images

    Returns:
        tensor_batch: prepared tensors
    """
    tensors = [preprocessors(img) for img in imgs]
    tensor_batch = torch.stack(tensors, dim=0)

    return tensor_batch
