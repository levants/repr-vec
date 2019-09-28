"""
Created on Sep 27, 2019

Index extracted vectors

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mimetypes
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from repr.models.vecs import Vectorizer
from utils.logging import logger

# Static data
IMG_EXTS = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))


def _dump_data(dst: str, reprs: list, verbose: bool = True):
    """
    Serialize vectors in file
    Args:
        dst: destination file path
        reprs: vectors to serialize
        verbose: logging flag
    """
    with open(dst, 'wb') as vecs:
        pkl.dump(reprs, vecs)
        logger.print_texts(verbose, f'Saved len(full_dicts) = {len(full_dicts)}')


def _lead_data(vectors_file: str, verbose: bool = True):
    """
    Read serialized vectors
    Args:
        vectors_file: directory to store vectors
        verbose: logging flag

    Returns:
        images_dict: list of dictionaries of vectorized images
    """
    with open(vectors_file, 'rb') as vecs:
        images_dict = pkl.load(vecs)
        logger.print_texts(verbose, f'{len(images_dict)} vectors are extracted')

    return images_dict


def _extract(model: Vectorizer, paths: list) -> np.ndarray:
    """
    Extract vector from image
    Args:
        model: representation extractor model
        paths: paths of images

    Returns:
        vec: extracted vector
    """
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
        vec = model(img)
        yield vec


def index_dir(model: Vectorizer, src: Path, dst: Path):
    """
    Index image representations
    Args:
        model: model for representation
        src: source directory of images
        dst: destination directory for indexing
    """
    paths = [pt for pt in src.iterdir() if pt.suffix in IMG_EXTS]
    vecs = list(_extract(model, paths))
    _dump_data(str(dst), vecs)
