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
from scipy.spatial.distance import cosine

from repr.models.encoders import Encoder
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
        logger.print_texts(verbose, f'Saved len(full_dicts) = {len(reprs)}')


def _load_data(vectors_file: str, verbose: bool = True):
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


def _encode(model: Encoder, path: str) -> tuple:
    """
    Extract vector from image
    Args:
        model: representation extractor model
        path: image path

    Returns:
        vec: vector from image
        img: original image
    """
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
    vec = model(img)

    return vec, img


def _encode_all(model: Encoder, paths: list) -> np.ndarray:
    """
    Extract vector from image
    Args:
        model: representation extractor model
        paths: paths of images

    Returns:
        vec: extracted vector
        path: image path
    """
    for path in paths:
        vec, _ = _encode(model, path)
        yield vec, path


def img_paths(src: Path) -> list:
    """
    Generate path list of images from directory
    Args:
        src: directory path

    Returns:
        path list of images
    """
    return [pt for pt in src.iterdir() if pt.suffix in IMG_EXTS]


def index_dir(model: Encoder, src: Path, dst: Path):
    """
    Index image representations
    Args:
        model: model for representation
        src: source directory of images
        dst: destination directory for indexing
    """
    paths = [pt for pt in src.iterdir() if pt.suffix in IMG_EXTS]
    vecs = list(_encode_all(model, paths))
    _dump_data(str(dst), vecs)


def _extract_img(vec, dbs_vecs) -> list:
    """
    Seach image in vectors
    Args:
        vec: source vector
        dbs_vecs: database vectors
        n_results: number of results

    Returns:
        dists: top results
    """
    dists = list()
    for vec2, path in dbs_vecs:
        dist = cosine(vec, vec2)
        dists.append((dist, path))
        dists = sorted(dists, key=lambda tup: tup[0])

    return dists


def search_img(vec, dbs_vecs, n_results: int = None) -> list:
    """
    Seach image in vectors
    Args:
        vec: source vector
        dbs_vecs: database vectors
        n_results: number of results

    Returns:
        dists: top results
    """
    dists = _extract_img(vec, dbs_vecs)
    dists = dists[:n_results] if n_results else dists

    return dists


def search_dir(model: Encoder, paths: list, index: Path, n_results: int = None) -> list:
    """
    Search files and extract
    Args:
        model: model for representation
        paths: path of images to seach
        index: index file
        n_results: number of results

    Returns:
        res_vecs: result images
    """
    res_vecs = list()
    src_vecs = [(_encode(model, path), path) for path in paths]
    dbs_vecs = _load_data(str(index))
    for (vec1, img), pt in src_vecs:
        dists = search_img(vec1, dbs_vecs, n_results=n_results)
        res_vecs.append((img, dists, pt))

    return res_vecs
