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
        self.preprocessors = preprocessors

    def forward(self, *imgs: np.ndarray) -> np.ndarray:
        tensor_batch = torch.stack([self.preprocessors(img) for img in imgs], dim=0)
        with torch.no_grad():
            tensor_batch = tensor_batch.cuda() if self.gpu and torch.cuda.is_available() else tensor_batch
            repr_tens = self.model.vec(tensor_batch) if hasattr(self.model, 'vec') else self.model(tensor_batch)
            repr_vec = repr_tens.cpu().detach().numpy()

        return repr_vec

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
