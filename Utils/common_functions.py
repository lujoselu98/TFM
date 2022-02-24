import itertools
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import sklearn.model_selection

from Utils import fixed_values
from Utils import paths


def load_data(dataset: str) -> Tuple[np.array, pd.DataFrame, pd.Series]:
    """
    Function to load data
    :param dataset: dataset identifier [CC, CDCOR, FFT]
    :return: features_names, data matrix, label vector
    """
    if dataset == 'CC':
        data_path = paths.CC_DATA_PATH
    elif dataset == 'DCOR':
        data_path = paths.CDCOR_DATA_PATH
    elif dataset == 'FFT':
        data_path = paths.FFT_DATA_PATH
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    X = pd.read_pickle(f"{data_path}/X.pickle")
    y = pd.read_pickle(f"{data_path}/y.pickle")
    tt = X.columns

    if dataset == 'FFT':
        X.columns = X.columns.astype('str')

    return tt, X, y


def get_fold(X: pd.DataFrame, y: pd.Series, idx_external: int, idx_internal: Optional[int] = None,
             strategy: Optional[str] = 'kfold') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get X_train, X_test, y_train, y_test corresponding to the fold
    :param X: Whole data matrix
    :param y: Whole label vector
    :param idx_external: idx of the fold of external division
    :param idx_internal: idx of the fold of internal division
    :param strategy: The strtegy to make the fold of external division ['kfold', 'randomsplit']
    :return:  X_train, X_test, y_train, y_test
    """

    assert strategy in ['kfold', 'randomsplit']

    if strategy == 'kfold':
        external_cv = sklearn.model_selection.StratifiedKFold(n_splits=fixed_values.EXTERNAL_SPLITS, shuffle=True,
                                                              random_state=0)
    else:
        external_cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=fixed_values.EXTERNAL_SPLITS_SHUFFLE,
                                                                     test_size=fixed_values.EXTERNAL_TEST_SIZE,
                                                                     random_state=0)
    splits = list(external_cv.split(X, y))
    index_train, index_test = splits[idx_external]

    if idx_internal is None:
        return X.iloc[index_train], X.iloc[index_test], y.iloc[index_train], y.iloc[index_test]
        #      X_train            , X_test            , y_train            , y_test

    if idx_internal is not None:
        # Dentro del interno la X y la y son los train del externo 9/10 del total
        X_int, y_int = X.iloc[index_train], y.iloc[index_train]

        # Usamos el idx externo para el random del shuffle del interno
        internal_cv = sklearn.model_selection.StratifiedKFold(n_splits=fixed_values.INTERNAL_SPLITS, shuffle=True,
                                                              random_state=idx_external)
        splits = list(internal_cv.split(X_int, y_int))
        index_train, index_test = splits[idx_internal]

        return X_int.iloc[index_train], X_int.iloc[index_test], y_int.iloc[index_train], y_int.iloc[index_test]


def get_all_permutations(dictionary: Dict) -> List[Dict]:
    """
        Return all posible combinations of (key,value) from a dict

    """
    keys, values = zip(*dictionary.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _print_functions():
    """
        Just print defined functions docstring
    """
    print(load_data.__doc__)
    print(get_fold.__doc__)
    print(get_all_permutations.__doc__)


if __name__ == '__main__':
    _print_functions()
