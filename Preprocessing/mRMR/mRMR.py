"""
    All the functions related to mRMR preprocessing
"""
from typing import Optional, List

import dcor
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as nbtqdm


def calculate_mRMR(X_train: pd.DataFrame, y_train: pd.Series, features_number: int, use_tqdm: Optional[bool] = False,
                   tqdm_desc: Optional[str] = '', is_notebook: Optional[bool] = False) -> List[int]:
    """

    :param X_train: Data matrix (features as columns, patterns as rows)
    :param y_train: Label matrix
    :param features_number: Number of features to select
    :param use_tqdm: use of tqdm progress bar
    :param tqdm_desc: description of to set into tqdm progress bar
    :param is_notebook: use of tqdm in notebook to allow calls fron notebook in nice format
    :return: Indexes of the selected features
    """
    # Set types and shapes
    X_train_t = X_train.T
    y_train = y_train.astype('float')

    # Initialize
    relevance_vector = dict()
    accumulated_redundancy = dict()
    candidates = X_train_t.index.to_list()

    # Relevance
    for idx in candidates:
        row_vals = X_train_t.loc[idx]
        relevance_vector[idx] = dcor.distance_correlation(row_vals, y_train, method=dcor.DistanceCovarianceMethod.AVL)
        accumulated_redundancy[idx] = 0

    # Auxiliar for max relevance
    v = list(relevance_vector.values())
    k = list(relevance_vector.keys())

    # get first from relevance
    last_feature_selected = k[v.index(max(v))]
    selected_features_index = [last_feature_selected]
    candidates.remove(last_feature_selected)

    # Iteration loop for selection
    if use_tqdm:
        if is_notebook:
            pbar = nbtqdm(total=features_number - 1, desc=tqdm_desc)
        else:
            pbar = tqdm(total=features_number - 1, desc=tqdm_desc)

    while len(selected_features_index) < features_number:
        maximum = -np.inf
        new_feature_index = 0
        for idx in candidates:
            actual_vals = X_train_t.loc[idx]
            last_vals = X_train_t.loc[last_feature_selected]
            relevance = relevance_vector[idx]
            accumulated_redundancy[idx] += dcor.distance_correlation(actual_vals, last_vals,
                                                                     method=dcor.DistanceCovarianceMethod.MERGESORT)
            redundancy = accumulated_redundancy[idx] / len(selected_features_index)
            mrmr = relevance - redundancy
            if mrmr > maximum:
                maximum = mrmr
                new_feature_index = idx
        last_feature_selected = new_feature_index
        selected_features_index.append(last_feature_selected)
        candidates.remove(last_feature_selected)
        if use_tqdm:
            pbar.update(1)
    if use_tqdm:
        pbar.close()
    # return X_train[selected_features_index], X_test[selected_features_index], selected_features_index
    return selected_features_index
