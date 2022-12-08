"""
    Used for utility functions for pandas dataframe
"""
import json
from typing import Any

import numpy as np


def constant_size(signal_values: np.ndarray, threshold: int = 1) -> int:
    """

    Calculate the total size of the constant part of the signal

    :param signal_values: signal to calculate constant part size
    :param threshold: param to set which size of the signal parts takes into account
    :return: the len of the sum of all constant parts on signal
    """
    # noinspection PyTypeChecker
    return np.sum(
        [
            x.size
            for x in np.split(
                signal_values, np.where(np.diff(signal_values) != 0)[0] + 1
            )
            if x.size > threshold
        ]
    )


def extract_dict(dict_string: str, key: Any) -> Any:
    """
    Extract a key from dict in string mode, util for extract a key from a dict column in pandas

    :param dict_string:  dictionary
    :param key: key
    :return: value of the key in dictionary
    """

    params_dict = json.loads(
        dict_string.replace("'", ";").replace('"', "'").replace(";", '"')
    )
    return params_dict[key]
