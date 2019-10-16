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


class Encoder(object):
    """Model wrapper for embedding extraction"""

    def __init__(self, model: nn.Module, preprocessors: callable, body: nn.Module = None, head: nn.Module = None,
                 gpu: bool = False):
        super(Encoder, self).__init__()
        eval_model = model.eval()
        self.body = nn.Sequential(*list(model.children())[:-2]).eval() if body is None else body.eval()
        self.head = nn.Sequential(*list(model.children())[-2:]) if head is None else head.eval()
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

    def _preprocess(self, *imgs: np.ndarray) -> torch.Tensor:
        """
        Process input data before fast-forward call
        Args:
            *imgs: images to process

        Returns:
            tensor to run
        """
        tensor_batch = torch.stack([self.preprocessors(img) for img in imgs], dim=0)
        tensor_batch = tensor_batch.cuda() if self.gpu and torch.cuda.is_available() else tensor_batch

        return tensor_batch

    def _run_layer(self, mpart: nn.Module, *imgs: np.ndarray):
        """
        Runs specific part of model
        Args:
            mpart: model part
            *imgs: input images

        Returns:
            repr_vec: representation extracted from input
        """
        tensor_batch = self._preprocess(*imgs)
        with torch.no_grad():
            repr_tens = mpart(tensor_batch)
            repr_vec = self.to_np(repr_tens)

        return repr_vec

    def slicer(self, *imgs: np.ndarray):
        """
        Extract body from data
        Args:
            *imgs: images to process

        Returns:
            repr_vec: extracts before representation layer from input data batch
        """
        return self._run_layer(self.body, *imgs)

    def vhead(self, *imgs: np.ndarray):
        """
        Represent head from body
        Args:
            *imgs: images to process

        Returns:
            repr_vec: extracts before representation layer from input data batch
        """
        return self._run_layer(self.head, *imgs)

    def forward(self, *imgs: np.ndarray) -> np.ndarray:
        return self._run_layer(self.extr, *imgs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
