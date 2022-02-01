"""
Functions to transformate the original data (cleaned) to data that we pass to the classifier. [cc(uc fhr), cdcor(uc, fhr), fft(uc, fhr)]

"""
from typing import List

import numpy as np
import pandas as pd
from dcor import distance_correlation, DistanceCovarianceMethod


def calc_lags(minutes: int = 5) -> List[int]:
    """

    Returns the lag for a given number of minutes

    :param minutes: minutes of max lag
    :return: all posible lags
    """
    N = minutes * 4 * 60

    return [k - N + 1 for k in range(0, 2 * N - 2 + 1)]


def cc(x: pd.Series, y: pd.Series, lag: int) -> float:
    """

    Nan save (pandas) lagged cross correlation between two signals

    :param x: first signal not lagged
    :param y: second signal lagged
    :param lag: lag to lag second signal
    :return: lagged cross correlation
    """
    return x.corr(y.shift(lag))


def shift(signal: np.ndarray, lag: int) -> np.ndarray:
    """

    Return the input signal lagged

    :param signal: array of points of signal
    :param lag: amount to shift the signal in points
    :return: shifted signal by lag points
    """
    lagged_signal = np.roll(signal, lag)
    if lag < 0:
        lagged_signal[lag:] = np.nan
    else:
        lagged_signal[:lag] = np.nan
    return lagged_signal


def dcor(x: np.ndarray, y: np.ndarray) -> float:
    """

    NaN save implementation of dcor.distance_correlation

    :param x: array of points of first signal
    :param y: array of points of second signal
    :return: distance_correlation between x and y
    """

    _x = x.copy()
    _y = y.copy()

    NaNs = np.logical_or(np.isnan(_x), np.isnan(_y))
    _x = _x[~NaNs]
    _y = _y[~NaNs]

    return distance_correlation(_x, _y, method=DistanceCovarianceMethod.MERGESORT)
