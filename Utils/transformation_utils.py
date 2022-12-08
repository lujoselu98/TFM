"""
Functions to transformate the original data (cleaned) to data that we pass to the classifier. [cc(uc fhr), cdcor(uc, fhr), fft(uc, fhr)]

"""
from typing import List

import joblib
import numpy as np
import pandas as pd
from dcor import DistanceCovarianceMethod, distance_correlation


def calc_lags(minutes: int = 5) -> list[int]:
    """

    Returns the lag for a given number of minutes

    :param minutes: minutes of max lag
    :return: all posible lags
    """
    N = minutes * 4 * 60

    return [k - N + 1 for k in range(0, 2 * N - 2 + 1)]


def cc(x: pd.Series, y: pd.Series, lag: int, min_num_points: int) -> float:
    """

    Nan save (pandas) lagged cross correlation between two signals

    :param x: first signal not lagged
    :param y: second signal lagged
    :param lag: lag to lag second signal
    :param min_num_points: mínimo numero de puntos para considerar la correlación válida
    :return: lagged cross correlation
    """
    return x.corr(y.shift(lag), min_periods=min_num_points)


def dcor(x: np.ndarray, y: np.ndarray, min_num_points: int) -> float:
    """

    NaN save implementation of dcor.distance_correlation

    :param x: array of points of first signal
    :param y: array of points of second signal
    :param min_num_points: mínimo numero de puntos para considerar la correlación válida
    :return: distance_correlation between x and y
    """

    _x = x.copy()
    _y = y.copy()

    NaNs = np.logical_or(np.isnan(_x), np.isnan(_y))
    if sum(~NaNs) <= min_num_points:
        return np.nan

    _x = _x[~NaNs]
    _y = _y[~NaNs]

    return distance_correlation(_x, _y, method=DistanceCovarianceMethod.MERGESORT)


def get_freqs(N: int, sample_freq: float = 4) -> np.ndarray:
    """

    Function to get the freq till nyquist limit

    :param N: size of signal in data points
    :param sample_freq: sample freq of the signal
    :return: array of freqs
    """

    return np.fft.rfftfreq(N, d=1.0 / sample_freq)


def nan_save_fft(signal: pd.Series, freqs: np.ndarray) -> list[float]:
    """
        Nan save fft implementation
    :param signal: given signal data
    :param freqs: freqs to calculate the fft
    :return: NaN save fft of signal for the given freq
    """
    N = len(signal)

    n = np.arange(N)

    nan_signal = np.isnan(signal)
    signal_nans = signal[~nan_signal]

    def parrallel_k(
        p_k: int, p_nan_signal: np.ndarray, p_signal_nans: pd.Series
    ) -> np.ndarray:
        """
        parallel part of function
        """
        nan_exp = np.exp(-2j * np.pi * p_k * n / N)[~p_nan_signal]
        return np.sum(p_signal_nans * nan_exp)

    fft = joblib.Parallel(n_jobs=8)(
        joblib.delayed(parrallel_k)(k, nan_signal, signal_nans)
        for k in range(len(freqs))
    )

    return (N / np.sum(~nan_signal)) * np.abs(fft)
