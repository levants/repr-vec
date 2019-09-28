"""
Created on Sep 27, 2019

Models for image representation vector generation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn


class Vectorizer(object):
    """Model wrapper for embedding extraction"""

    def __init__(self, model: nn.Module, preprocessors: callable, gpu: bool = False):
        super(Vectorizer, self).__init__()
        eval_model = model.eval()
        self.gpu = gpu and torch.cuda.is_available()
        self.model = eval_model.cuda() if self.gpu and torch.cuda.is_available() else eval_model
        self.extr = self.model.vec if hasattr(self.model, 'vec') else self.model
        self.preprocessors = preprocessors

    @staticmethod
    def to_np(src_tens: torch.Tensor) -> np.ndarray:
        """
        Converts tensor to array
        Args:
            src_tens: tensor to convert

        Returns:
            converted array
        """
        return src_tens.cpu().detach().numpy()

    def forward(self, *imgs: np.ndarray) -> np.ndarray:
        tensor_batch = torch.stack([self.preprocessors(img) for img in imgs], dim=0)
        with torch.no_grad():
            tensor_batch = tensor_batch.cuda() if self.gpu and torch.cuda.is_available() else tensor_batch
            repr_tens = self.extr(tensor_batch)
            repr_vec = self.to_np(repr_tens)

        return repr_vec

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
