"""
Created on Sep 10, 2019

Utility module for working on files

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from pathlib import Path


def dump(src_files: list, dst_dir: Path, verbose: bool = False):
    """
    Writes files to destination
    Args:
        src_files: source files to dump
        dst_dir: destination directory
        verbose: logging flag
    """
    for idx, src_file in enumerate(src_files):
        src_file_txt = str(src_file)
        if src_file.exists() and os.path.getsize(src_file_txt) > 100:
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file_txt, str(dst_file))
            if verbose: print(f'{idx} data if {len(src_files)} is processed')
