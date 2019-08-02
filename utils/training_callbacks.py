"""
Created on Aug 01, 2019

Training callbacks for logging progress and best model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from fastai.vision import *


class CSVLoggeerLine(callbacks.CSVLogger):
    """Implementation of CSVLogger with flush each epoch"""

    def __init__(self, learn: Learner, file_fir: Path = None, filename: str = 'history', append: bool = False,
                 flush_epoch: bool = True):
        super(CSVLoggeerLine, self).__init__(learn, filename=filename, append=append)
        self.flush_epoch = flush_epoch
        if file_fir:
            self.path = file_fir / f'{filename}.csv'

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        super(CSVLoggeerLine, self).on_epoch_end(epoch, smooth_loss, last_metrics, **kwargs)
        if self.flush_epoch:
            self.file.flush()


def init_callbacs(monitor: str = 'accuracy', pref: str = None, append: bool = False, flush_epoch: bool = True):
    """
    Initialize callbacks for training
    Args:
        monitor: monitor function
        pref: prefix for logger
        append: append to logging file
        flush_epoch: flush each epoch log

    Returns:
        lrn_callbacks: callbacks list
    """
    set_pref = pref if pref and pref.strip() else time.strftime("%Y-%m-%d-%H-%M-%S")
    file_pref = f'best-{set_pref}'
    hist_pref = f'history-{set_pref}'
    lrn_callbacks = [partial(callbacks.SaveModelCallback, every='improvement', monitor=monitor, name=file_pref),
                     partial(CSVLoggeerLine, filename=hist_pref, append=append, flush_epoch=flush_epoch)]

    return lrn_callbacks
