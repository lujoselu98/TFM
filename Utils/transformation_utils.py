"""
Functions to transformate the original data (cleaned) to data that we pass to the classifier. [cc(uc fhr), cdcor(uc, fhr), fft(uc, fhr)]

"""
import pandas as pd
from typing import List

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
