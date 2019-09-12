"""
Created on Sep 10, 2019

Utility module for working on files

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
from pathlib import Path


def dump(src_files: list, dst_dir: Path, func_valid: callable = lambda x: x, verbose: bool = False):
    """
    Writes files in to the destination directory
    Args:
        src_files: source files to dump
        dst_dir: destination directory
        func_valid: validation function for source file
        verbose: logging flag
    """
    for idx, src_file in enumerate(src_files):
        src_file_txt = str(src_file)
        if func_valid(src_file_txt):
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file_txt, str(dst_file))
            if verbose: print(f'{idx} data if {len(src_files)} is processed')
