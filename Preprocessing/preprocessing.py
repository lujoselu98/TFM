"""
    Functions to use during experiments to load preprocessin data from files
"""
import pickle
from typing import Tuple, Optional

import numpy as np

from Utils import paths, fixed_values


def load_preprocess(dataset: str, preprocess: str, idx_external: int, idx_internal: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray]:
    """

    Function to load a preprocessed dataset

    :param dataset: Dataset to load
    :param preprocess: Preprocess to load
    :param idx_external: idx of external division
    :param idx_internal: idx of internal division
    :return: X_train and X_test from file given the parameters
    """

    if preprocess == 'mRMR':
        path = paths.MRMR_PATH
    elif preprocess == 'PCA':
        path = paths.FPCA_PATH
    elif preprocess == 'PLS':
        path = paths.PLS_PATH
    else:
        raise ValueError(f"unknown preprocess: {preprocess}")

    file_f_string = "{PATH}/{dataset}_{preprocess}_{idx_external}{idx_internal}"

    idx_internal_s = ''
    if idx_internal is not None:
        idx_internal_s = '_' + str(idx_internal)

    pickle_path = file_f_string.format(PATH=path, data=dataset, preprocess=preprocess, idx_external=idx_external,
                                       idx_internal=idx_internal_s)

    with open(f"{pickle_path}_train.pickle", 'rb') as f:
        X_train_pre = pickle.load(f)

    with open(f"{pickle_path}_test.pickle", 'rb') as f:
        X_test_pre = pickle.load(f)

    return X_train_pre, X_test_pre


def get_features(X_train_pre: np.ndarray, X_test_pre: np.ndarray,
                 features_number: Optional[int] = fixed_values.MAX_DIMENSION) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selected a fixed number of features from X_train and X_test

    :param X_train_pre: Data Matrix Preprocessed of train patterns
    :param X_test_pre:  Data Matrix Preprocessed of test patterns
    :param features_number: Feature number to select
    :return:a fixed number of features from X_train_pre and X_test_pre
    """
    return X_train_pre[:, :features_number], X_test_pre[:, :features_number]
    #      X_train,                          X_test
