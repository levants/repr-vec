"""
Created on Jun 11, 2018

Logger for training and prediction

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from utils.logging.logger import *


class LoggerTest(unittest.TestCase):
    """Test class for model configuration"""

    def setUp(self):
        self.line_seps1 = '=' * CHAR_NUM
        self.line_seps2 = '#' * CHAR_NUM

    def test_line_separators(self):
        """Tests line separator functions"""
        assert len(self.line_seps1) == len(self.line_seps2)
        print('len(LINE_SEP) = ', len(LINE_SEP))
        print('len(line_seps1) = ', len(self.line_seps1))
        print('len(line_seps2) = ', len(self.line_seps2))
        print('line_seps1 - ', self.line_seps1)
        print('line_seps2 - ', self.line_seps2)
        print()
        print()
        print()
        print()
        separate_lines(lines=6)
