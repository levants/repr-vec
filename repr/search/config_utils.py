"""
Created on Oct 12, 2019

Configuration utilities for indexing and searching in directory

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from path_utils import data_path
from repr.models.resnet import resnet50


def configure() -> argparse.Namespace:
    """
    Configuration parameters
    Returns:
        config: configuration parameters namespace
    """
    parser = argparse.ArgumentParser('Configuration for indexing and extraction directories')
    # Model configuration
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        default=resnet50.__name__,
                        help='Backbone model architecture for encoder')
    # Runtime configuration
    parser.add_argument('--search',
                        dest='search',
                        action='store_true',
                        help='Search or index directory')
    parser.add_argument('--slice',
                        dest='slice',
                        action='store_true',
                        help='Slice or index directory')
    # Files configuration
    parser.add_argument('--path',
                        dest='path',
                        type=str,
                        default=str(data_path() / 'label_engine'),
                        help='Root directory path')
    parser.add_argument('--weights',
                        dest='weights',
                        type=str,
                        default=str(data_path() / 'label_engine' / 'weights' / 'a-rot2-f2.pth'),
                        help='Pre-ytrained weights file path')
    parser.add_argument('--src',
                        dest='src',
                        type=str,
                        default=str(data_path() / 'label_engine' / 'cropped_base' / 'train'),
                        help='Source directory to index')
    parser.add_argument('--dst',
                        dest='dst',
                        type=str,
                        default=str(data_path() / 'label_engine' / 'db' / 'vectors.pkl'),
                        help='Directory to store indexed vectors')
    parser.add_argument('--qur',
                        dest='qur',
                        type=str,
                        default=str(data_path() / 'label_engine' / 'query_cropped'),
                        help='Query files directory')
    parser.add_argument('--gt',
                        dest='gt',
                        type=str,
                        default=str(data_path() / 'label_engine' / 'gt.csv'),
                        help='Ground true files relation CSV data')
    # Configure indexer
    parser.add_argument('--bs',
                        dest='bs',
                        type=int,
                        default=1,
                        help='Batch size for indexer')
    parser.add_argument('--n_results',
                        dest='n_results',
                        type=int,
                        default=10,
                        help='Number of top results')
    # Configure logging
    parser.add_argument('--step',
                        dest='step',
                        type=int,
                        default=10,
                        help='Step to log indexer progress')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='Logging flag')
    config = parser.parse_args()

    return config
