"""
Created on Jul 25, 2019

Data utility module for rotation classifier

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from fastai.vision import *

# Extensions
_IMG_EXTS_1 = ['.jpg', '.jpeg', '.png']
_IMG_EXTS = set(_IMG_EXTS_1 + [ext.upper() for ext in _IMG_EXTS_1])

# Rotation angles
_CLASSES = [0, 90, 180, 270]


def init_transforms(size: int = None):
    """
    Initializes additional transforms
    Args:
        size: size of input image

    Returns:
        additional transformations
    """
    xtr_trsf_tail = [rand_crop(p=1.), crop_pad(), brightness(change=(0.1, 0.9), p=1.0),
                     contrast(scale=(0.5, 2.), p=1.), jitter(magnitude=-0.2, p=1.),
                     symmetric_warp(magnitude=(-0.2, 0.2), p=1.), rotate(degrees=(-15, 15), p=1.0),
                     zoom(scale=1.78, p=1.), cutout(n_holes=(1, 4), length=(10, 160), p=1.),
                     squish(scale=(0.68, 1.38))]
    xtr_trsf = rand_resize_crop(size) + xtr_trsf_tail if size and size > 0 else xtr_trsf_tail

    return xtr_trsf


def make_dirs(dst_dir: Path) -> dict:
    """
    Make class directories
    Args:
        dst_dir: root destination directory

    """
    dest_dirs = dict()
    for cl in _CLASSES:
        dr = dst_dir / str(cl)
        dr.mkdir(exist_ok=True)
        dest_dirs[str(cl)] = dr

    return dest_dirs


def _init_rotation_matrix(h: int = 224, w: int = 224, tr_mx: dict = None) -> dict:
    """
    Rotation matrices
    Args:
        h: height of input
        w: width of input
        tr_mx: existing transformation (rotation matrices)

    Returns:
        ms: rotation matrices
    """
    if tr_mx is None:
        center = (w / 2, h / 2)
        ms = {str(cl): cv2.getRotationMatrix2D(center, cl, 1) for cl in _CLASSES if cl != 0}
        ms['0'] = None
    else:
        ms = tr_mx

    return ms


def _read_resize(p: str, h: int, w: int, interpolation: int) -> np.ndarray:
    """
    Read and resize image
    Args:
        p: path to image
        h: height to resize
        w: width to resize
        interpolation: interpolation for resize

    Returns:
        img: resized image
    """
    orig_img = cv2.imread(str(p), cv2.IMREAD_ANYCOLOR)
    if orig_img is None or len(orig_img.shape) < 2 or orig_img.shape[0] < 10 or orig_img.shape[1] < 10:
        img = None
    else:
        img = cv2.resize(orig_img, (w, h), interpolation=interpolation)

    return img


def generate_classes(src_dir: Path, dst_dir: Path, h: int = 224, w: int = 224, tr_mx: dict = None,
                     interpolation: int = cv2.INTER_LINEAR):
    """
    Generate rotation data-set for classification
    Args:
        src_dir: source directory
        dst_dir: destination directory
        h: height of image
        w: width of image
        tr_mx: rotation matrices
        interpolation: interpolation for resize

    """
    dest_dirs = make_dirs(dst_dir)
    img_dict = {str(p.name): _read_resize(p, h, w, interpolation) for p in src_dir.iterdir() if p is not None and \
                p.suffix in _IMG_EXTS}
    ms = _init_rotation_matrix(h=h, w=w, tr_mx=tr_mx)
    for k, v in img_dict.items():
        for d, m in ms.items():
            mod = v if d == '0' else cv2.warpAffine(v, m, (h, w))
            dst = dest_dirs[d]
            cv2.imwrite(str(dst / k), mod)


def _add_tests(src_root: Path, dst_root: Path, src_dirs: list, dst_dirs: list, tst_dir: str = None):
    """
    Add test directory to source and destination paths
    Args:
        src_root: source root
        dst_root: destination root
        src_dirs: source directories
        dst_dirs: destination directories
        tst_dir: test directory name

    """
    if tst_dir:
        src_dirs += [src_root / tst_dir]
        dst_dirs += [dst_root / tst_dir]


def generate_data(src_root: Path, dst_root: Path, h: int = 224, w: int = 224, tr_dir: str = 'train',
                  val_dir: str = 'valid', tst_dir: str = None):
    """
    Generate rotation data-set
    Args:
        src_root: source root directory
        dst_root: destination root directory
        h: height of image
        w: width of image
        tr_dir: training directory
        val_dir: validation directory
        tst_dir: test directory

    """
    src_dirs = [src_root / tr_dir, src_root / val_dir]
    dst_dirs = [dst_root / tr_dir, dst_root / val_dir]
    _add_tests(src_root, dst_root, src_dirs, dst_dirs, tst_dir=tst_dir)
    tr_mx = _init_rotation_matrix(h=h, w=w)
    for idx, src_dir in enumerate(src_dirs):
        if src_dir.exists():
            dst_dir = dst_dirs[idx]
            dst_dir.mkdir(exist_ok=True)
            generate_classes(src_dir, dst_dir, h=h, w=w, tr_mx=tr_mx)


def label_and_folder(src_root: Path, dst_root: Path, h: int = 224, w: int = 224, train: PathOrStr = 'train',
                     valid: PathOrStr = 'valid', valid_pct: float = None):
    """
    Label and put data in folders
    Args:
        src_root: source root directory
        dst_root: destination root directory
        h: height of image
        w: width of image
        train: training directory
        valid: validation directory
        valid_pct: validation percentage

    """
    if valid_pct is None:
        generate_data(src_root, dst_root, h=h, w=w, tr_dir=train, val_dir=valid)
    else:
        generate_classes(src_root, dst_root, h=h, w=w, tr_mx=None)


def init_databunch(dst_root: Path, train: PathOrStr = 'train', valid: PathOrStr = 'valid', valid_pct=None,
                   seed: int = None, size: int = None, classes: Collection = None, bs: int = 64, val_bs: int = None,
                   ds_tfms: Optional[TfmList] = None, num_workers=None, **kwargs: Any) -> ImageDataBunch:
    """
    Initializes image data bunch
    Args:
        dst_root: destination root directory
        train: training directory
        valid: validation directory
        valid_pct: validation percentage
        seed: random seed
        size: input size
        classes: classes for training
        bs: batch size
        val_bs: validation batch size
        ds_tfms: data transformation
        num_workers: number of worker threads
        **kwargs: additional arguments

    Returns:
        initialized image data bunch
    """
    return ImageDataBunch.from_folder(dst_root, train=train, valid=valid, valid_pct=valid_pct, seed=seed,
                                      classes=classes, bs=bs, val_bs=val_bs, size=size, ds_tfms=ds_tfms,
                                      num_workers=num_workers, **kwargs)


def rotation_databunch(src_root: Path, dst_root: Path, h: int = 224, w: int = 224, size: int = None,
                       train: PathOrStr = 'train', valid: PathOrStr = 'valid', valid_pct=None, seed: int = None,
                       classes: Collection = None, bs: int = 64, val_bs: int = None, ds_tfms: Optional[TfmList] = None,
                       **kwargs: Any) -> ImageDataBunch:
    """
    Split data to directories
    Args:
        src_root: source root directory
        dst_root: destination root directory
        h: height of image
        w: width of image
        size: input size
        train: training directory
        valid: validation directory
        valid_pct: validation percentage
        seed: random seed
        classes: classes for training
        bs: batch size
        val_bs: validation batch size
        ds_tfms: data transformation
        **kwargs: additional arguments

    Returns:
        data_bunch: image data bunch for training / validation
    """
    dst_root.mkdir(exist_ok=True)
    label_and_folder(src_root, dst_root, h=h, w=w, train=train, valid=valid, valid_pct=valid_pct)
    im_size = (h + w) / 2 if size is None else size
    data_bunch = init_databunch(dst_root, train=train, valid=valid, valid_pct=valid_pct, seed=seed, size=im_size,
                                classes=classes, bs=bs, val_bs=val_bs, ds_tfms=ds_tfms, **kwargs)

    return data_bunch
