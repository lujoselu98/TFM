"""
    Used for utility functions for pandas dataframe
"""
import numpy as np


def constant_size(signal_values: np.array) -> int:
    """

    Calculate the total size of the constant part of the signal

    :param signal_values: signal to calculate constant part size
    :return: the len of the sum of all constant parts on signal
    """
    return np.sum([x.size
                   for x in np.split(signal_values, np.where(np.diff(signal_values) != 0)[0] + 1)
                   if x.size > 1])[0]
