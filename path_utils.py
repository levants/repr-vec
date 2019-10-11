"""
Created on Jul 09, 2019

Utility module for data and model files

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path


def data_path():
    """
    Initialize data path
    Returns:
        data path
    """
    return Path(__file__).parent / 'data'


def onnx_path():
    """
    Initialize ONNX files directory path
    Returns:
        onnx_dir: ONNX directory path
    """
    data = data_path()
    onnx_dir = data / 'onnx'

    return onnx_dir
