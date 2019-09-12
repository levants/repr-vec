"""
Created on Feb 23, 2018

Logger for training and prediction

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

# Default parameters
CHAR_NUM = 46
LINE_SEP = '=' * CHAR_NUM


class TimerWrapper(object):
    """Timer wrapper object"""

    def __init__(self, verbose, func=None):
        super(TimerWrapper, self).__init__()
        self.verbose = verbose
        self.func = func if func else 'function'
        if self.verbose:
            self.start = time.clock()

    def timeit(self):
        """
        Logs timed data
        Returns:
            time_taken: time for line / lines executions
        """
        if self.verbose:
            time_taken = time.clock() - self.start
            print_texts(self.verbose, 'Time taken for ', self.func, ' is - ', time_taken)
        else:
            time_taken = None

        return time_taken


def _is_verbose(flags):
    """
    Validates if flags are configured for logging
    Args:
        flags: training and validation configuration parameters

    Returns:
        if logging is set
    """
    return hasattr(flags, 'verbose') and flags.verbose


def _print_texts(*texts):
    """
    Prints passed texts
    Args:
        *texts: array of strings to print
    """
    out_text = ''.join(str(text) for text in texts)
    print(out_text)


def print_directly(flags, *texts):
    """
    Prints passed object directly
    Args:
        flags: configuration parameters
        *texts: array of strings to print
    """
    if _is_verbose(flags):
        _print_texts(*texts)


def print_texts(verbose, *texts):
    """
    Prints passed object directly
    Args:
        verbose: logging flag
        *texts: array of strings to print
    """
    if verbose:
        _print_texts(*texts)


def log_array(flags, array_datas, index_labels):
    """
    Logs passed array
    Args:
        flags: configuration flags
        array_datas: array to log
        index_labels: dictionary of indices and labels
    """
    if _is_verbose(flags) and array_datas is not None and len(array_datas) > 0:
        for array_data in array_datas:
            for i in range(len(array_data)):
                if i < len(index_labels):
                    print(index_labels[i], array_data[i])


def log(flags, *_messages):
    """
    Logs passed message
    Args:
        flags: configuration flags
        *_messages: messages to log
    """
    if _is_verbose(flags) and _messages is not None and len(_messages) > 0:
        final_message = ''
        for _message in _messages:
            final_message += str(_message)
        print(final_message)


def describe_model(verbose, model):
    """
    Prints model details if verbose flag is on
    Args:
        verbose: flags to print model
        model: network model
    """
    if verbose:
        print(model)


def print_model(flags, model):
    """
    Prints passed model
    Args:
        flags: configuration parameters
        model: network model
    """
    if _is_verbose(flags):
        model.summary()


def _log_discriminator(lines=3):
    """
    Prints lines to distinguish between logs
    Args:
        lines: lines to be filled with special symbols
    """
    for _ in range(lines):
        print()
    for _ in range(lines):
        print('=================================')
    for _ in range(lines):
        print()


def separate_lines(lines=3):
    """
    Prints lines to distinguish between logs
    Args:
        lines: lines to be filled with special symbols
    """
    if lines > 0:
        _log_discriminator(lines=lines)


def sep_lines(flags, lines=2):
    """
    Separates lines with special characters
    Args:
        flags: configuration parameters
        lines: amount of lines to print
    """
    if _is_verbose(flags):
        for _ in range(lines):
            print(LINE_SEP)


def print_models(flags, *models):
    """
    Prints passed model
    Args:
        flags: configuration parameters
        *models: network models
    """
    if _is_verbose(flags) and models and len(models):
        for model in models:
            _log_discriminator()
            model.summary()
            _log_discriminator()


def log_generators(flags, train_generator, validation_generator, test_generator):
    """
    Logs generator values
    Args:
        flags: training flags
        train_generator: training data generator
        validation_generator: validation data generator
        test_generator: test data generator
    """
    if _is_verbose(flags):
        print(train_generator.samples, train_generator.batch_size)
        print(validation_generator.samples, validation_generator.batch_size)
        print(test_generator.samples, test_generator.batch_size)


def print_samples_amount(flags, step_per_epoch, val_per_epoch):
    """
    Prints training and validation samples
    Args:
        flags: training flags
        step_per_epoch: training steps per epochs
        val_per_epoch: validation steps per epochs
    """
    if _is_verbose(flags):
        print('step_per_epoch -', step_per_epoch, 'val_per_epoch -', val_per_epoch)


def print_training_history(flags, history):
    """
    Prints training history
    Args:
        flags: training flags
        history: training history
    """
    if _is_verbose(flags):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        for i in epochs:
            print('epoch - ', i + 1, ' val_acc - ', val_acc[i], ' loss - ', loss[i], ' val_loss - ', val_loss[i])


def log_evaluation(flags, model, scores):
    """
    Logs model final test evaluation result
    Args:
        flags: training flags
        model: network model
        scores: final test results
    """
    if _is_verbose(flags):
        for i in range(len(model.metrics_names)):
            print("Final test results - %s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))


def start_timer(verbose, func=None):
    """
    Starts timer service
    Args:
        verbose: logging flag
        func: function name

    Returns:
        initialized timer instance
    """
    return TimerWrapper(verbose, func=func)
