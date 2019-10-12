"""
Created on Sep 27, 2019

Index extracted vectors

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl

import cv2
from fastai.vision import *
from scipy.spatial.distance import cosine

from repr.models.encoders import Encoder
from repr.models.resnet import resnet_vec
from repr.search.config_utils import configure
from repr.search.input_utils import init_transforms
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


def load_dbvecs(vectors_file: str, verbose: bool = True):
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


def _valid_img(img: np.ndarray, min_siz: int = 50) -> bool:
    """
    Validates image on minimum size
    Args:
        img: source image
        min_siz: minimum size requirement

    Returns:
        vld_img: validation on size
    """
    vld_img = img is not None and hasattr(img, 'shape')
    if vld_img:
        h, w = img.shape[:2]
        vld_img = h >= min_siz and w >= min_siz

    return vld_img


def _read_images(*paths: Path, min_siz: int = 50) -> tuple:
    """
    Validate and read images
    Args:
        *paths: paths of images
        min_siz: minimum image size to validate

    Returns:
        imgs: images from paths
        valid_paths: valid paths
    """
    imgs, valid_paths = list(), list()
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
        if _valid_img(img, min_siz=min_siz):
            imgs.append(img)
            valid_paths.append(path)

    return imgs, valid_paths


def _encode(model: Encoder, *paths: str, min_siz: int = 50) -> tuple:
    """
    Extract vector from image
    Args:
        model: representation extractor model
        path: image path
        min_siz: minimum image size

    Returns:
        vecs: vectors from images
        imgs: original images
        valid_paths: valid paths
    """
    imgs, valid_paths = _read_images(*paths, min_siz=min_siz)
    if valid_paths:
        vecs = model(*imgs)
    else:
        vecs = None

    return vecs, imgs, valid_paths


def _log_diff(img_diff: int, ivld_cnt: int, verbose: bool):
    if img_diff > 0:
        logger.print_texts(verbose, f'there are {ivld_cnt} invalid images')


def _encode_all(model: Encoder, *paths: list, min_siz: int = 50, verbose: bool = False, step: int = 100) -> np.ndarray:
    """
    Extract vector from image
    Args:
        model: representation extractor model
        paths: paths of images
        min_siz: minimum image size
        verbose: logging flag
        step: step to log after

    Returns:
        vec: extracted vector
        path: image path
    """
    ivld_cnt = 0
    for idx, path in enumerate(paths):
        vecs, _, valid_paths = _encode(model, *path, min_siz=min_siz)
        img_diff = len(path) - len(valid_paths)
        ivld_cnt += max(img_diff, 0)
        _log_diff(img_diff, ivld_cnt, verbose)
        if valid_paths:
            logger.print_texts(verbose and idx % step == 0,
                               f'{idx} data is processed, out of {len(paths)}, valid - {idx * len(path) - ivld_cnt}')
            yield vecs, valid_paths


def listify_results(vec_bts: list) -> list:
    """
    Flatten list of data
    Args:
        vec_bts: representation vectors batches

    Returns:
        flatten collection of representation vectors and paths
    """
    return [(vec, path) for vec_bt, path_bt in vec_bts for vec, path in zip(vec_bt, path_bt)]


def img_paths(src: Path) -> list:
    """
    Generate path list of images from directory
    Args:
        src: directory path

    Returns:
        path list of images
    """
    return [pt for pt in src.iterdir() if pt.suffix in IMG_EXTS]


def index_dir(model: Encoder, src: Path, dst: Path, min_siz: int = 50, bs: int = None, verbose: bool = False,
              step: int = 100):
    """
    Index image representations
    Args:
        model: model for representation
        src: source directory of images
        dst: destination directory for indexing
        min_siz: minimum image size
        bs: batch size
        verbose: logging flag
        step: step to log after
    """
    paths_list = [pt for pt in src.iterdir() if pt.suffix in IMG_EXTS]
    paths = [paths_list[i:i + bs] for i in range(0, len(paths_list), bs)]
    logger.print_texts(verbose, f'there are {len(paths)} images to index')
    vec_bts = list(_encode_all(model, *paths, min_siz=min_siz, verbose=verbose, step=step))
    vecs = listify_results(vec_bts)
    _dump_data(str(dst), vecs, verbose=verbose)


def _extract_img(vec: list, dbs_vecs: list, verbose: bool = False) -> list:
    """
    Seach image in vectors
    Args:
        vec: source vector
        dbs_vecs: database vectors
        verbose: logging flag

    Returns:
        dists: top results
    """
    timer = logger.start_timer(verbose, 'compare_images')
    dists = [(cosine(vec, vec2), path) for vec2, path in dbs_vecs]
    timer.timeit()
    timer = logger.start_timer(verbose, 'sort_result')
    dists = sorted(dists, key=lambda tup: tup[0])
    timer.timeit()

    return dists


def search_img(vec: list, dbs_vecs: list, n_results: int = None, verbose: bool = False) -> list:
    """
    Seach image in vectors
    Args:
        vec: source vector
        dbs_vecs: database vectors
        n_results: number of results
        verbose: logging flag

    Returns:
        dists: top results
    """
    dists = _extract_img(vec, dbs_vecs, verbose=verbose)
    dists = dists[:n_results] if n_results else dists

    return dists


def listify_dir(index: Path, verbose: bool = False):
    """
    Flatten serialized batches
    Args:
        index: path to serialized batches
        verbose: logging flag
    """
    bs_vecs = load_dbvecs(str(index))
    vecs = listify_results(bs_vecs)
    _dump_data(str(index), vecs, verbose=verbose)


def search_dir(model: Encoder, paths: list, db_vecs: list = None, index: Path = None, n_results: int = None,
               verbose: bool = False) -> tuple:
    """
    Search files and extract
    Args:
        model: model for representation
        paths: path of images to search
        db_vecs: database vectors
        index: index file
        n_results: number of results
        verbose: logging flag

    Returns:
        res_vecs: result images
    """
    res_vecs = list()
    src_vec_bts = list(_encode(model, path) for path in paths)
    src_vecs = [(vec, img, path) for vec_bt, img_bt, path_bt in src_vec_bts for vec, img, path in
                zip(vec_bt, img_bt, path_bt) if vec_bt is not None]
    logger.print_texts(verbose, f'len(src_vecs) = {len(src_vecs)} query images are vectorized')
    dbs_vecs = db_vecs if db_vecs else load_dbvecs(str(index))
    logger.print_texts(verbose, f'{len(dbs_vecs)} is loaded from disk')
    for idx, (vec1, img, pt) in enumerate(src_vecs):
        dists = search_img(vec1, dbs_vecs, n_results=n_results, verbose=verbose)
        res_vecs.append((img, dists, pt))
        logger.print_texts(verbose, f'result for {idx + 1} out of {len(src_vecs)} extracted , path - {pt}')

    return res_vecs, db_vecs


def encoder_model(arch: str = 'resnet50', weights: str = None) -> Encoder:
    """
    Initialize encoder model
    Args:
        arch: backbone model architecture for encoder
        weights: trained weights file path

    Returns:
        encoder: encoder model
    """
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(2048, 4))
    model = resnet_vec(arch, head=head, weights=weights)
    backbone = nn.Sequential(model[0], model[1][:-1])
    transforms = init_transforms(h=512, w=512, percnt=0.1, crop_center=True)
    encoder = Encoder(backbone, transforms, gpu=True)
    encoder.vec = lambda x: encoder(x)

    return encoder


if __name__ == '__main__':
    """Index and search full directory"""
    config = configure()
    path = Path(config.path)
    weights = Path(config.weights)
    src = Path(config.src)
    dst = Path(config.dst)
    qur = Path(config.qur)
    gt = Path(config.gt)
    encoder = encoder_model(arch=config.arch, weights=None)
    if config.search:
        paths = img_paths(qur)
        result_data, db_vecs = search_dir(encoder, paths, index=dst, n_results=config.n_results, verbose=config.verbose)
        print(f'result_data = {result_data} and db_vecs = {db_vecs}')
    else:
        index_dir(encoder, src, dst, bs=config.bs, verbose=config.verbose, step=config.step)
