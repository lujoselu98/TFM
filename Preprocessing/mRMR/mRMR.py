"""
    All the functions related to mRMR preprocessing
"""
import pickle
from typing import Optional, List

import dcor
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as nbtqdm

from Utils import common_functions, fixed_values, paths


def calculate_mRMR(X_train: pd.DataFrame, y_train: pd.Series, features_number: int, use_tqdm: Optional[bool] = False,
                   tqdm_desc: Optional[str] = '', is_notebook: Optional[bool] = False) -> List[int]:
    """

    :param X_train: Data matrix (features as columns, patterns as rows)
    :param y_train: Label matrix
    :param features_number: Number of features to select
    :param use_tqdm: use of tqdm progress bar
    :param tqdm_desc: description of to set into tqdm progress bar
    :param is_notebook: use of tqdm in notebook to allow calls fron notebook in nice format
    :return: Selected features names
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


def save_mRMR_indexes(dataset: str) -> None:
    """
        Save mRMR indexes into .txt files
    :param dataset: Dataset to calculate and save the indexes
    """
    tt, X, y = common_functions.load_data(dataset)
    for idx_external in range(fixed_values.EXTERNAL_SPLITS):
        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external)

        tqdm_desc = f"External fold {idx_external + 1}/{fixed_values.EXTERNAL_SPLITS} "
        selected_features_index = calculate_mRMR(X_train, y_train, features_number=fixed_values.MAX_DIMENSION,
                                                 use_tqdm=True, tqdm_desc=tqdm_desc)

        sel_features_file = f"{paths.MRMR_PATH}/{dataset}_sel_features_{idx_external}.txt"
        with open(sel_features_file, 'w') as f:
            f.write(str(selected_features_index))
        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal)

            tqdm_desc = f"External fold {idx_external + 1}/{fixed_values.EXTERNAL_SPLITS} " \
                        f"Internal fold {idx_internal + 1}/{fixed_values.INTERNAL_SPLITS}"
            selected_features_index = calculate_mRMR(X_train, y_train, features_number=fixed_values.MAX_DIMENSION,
                                                     use_tqdm=True, tqdm_desc=tqdm_desc)

            sel_features_file = f"{paths.MRMR_PATH}/{dataset}_sel_features_{idx_external}_{idx_internal}.txt"
            with open(sel_features_file, 'w') as f:
                f.write(str(selected_features_index))


def load_mRMR_indexes(dataset: str, idx_external: int, idx_internal: Optional[int] = None) -> List[int]:
    """
        Load indexes from .txt file, used to save the real values needed for experiments
    :param dataset: Dataset to load
    :param idx_external: External idx to load
    :param idx_internal: Internal idx to load
    :return: Read indexes
    """
    if idx_internal is None:
        mRMR_indexes_file = f"{paths.MRMR_PATH}/{dataset}_sel_features_{idx_external}.txt"

    else:  # idx_internal is not None
        mRMR_indexes_file = f"{paths.MRMR_PATH}/{dataset}_sel_features_{idx_external}_{idx_internal}.txt"

    with open(f'{mRMR_indexes_file}', 'r') as f:
        line = f.readline()
        mRMR_indexes = line.strip().replace("]", "").replace("[", "").replace("'", "").split(', ')
        if dataset != 'FFT':
            mRMR_indexes = [int(x) for x in mRMR_indexes]
        else:
            mRMR_indexes = [float(x) for x in mRMR_indexes]
    return mRMR_indexes


def save_mRMR(dataset: str) -> None:
    """
        Save the data of selected features by mRMR to use in experiments
    :param dataset: Dataset to use
    """
    tt, X, y = common_functions.load_data(dataset)
    for idx_external in range(fixed_values.EXTERNAL_SPLITS):
        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external)
        mRMR_indexes = load_mRMR_indexes(dataset, idx_external)

        X_train_mRMR = X_train[map(str, mRMR_indexes)].values
        X_test_mRMR = X_test[map(str, mRMR_indexes)].values

        pickle_file = f"{paths.MRMR_PATH}/{dataset}_mRMR_{idx_external}"

        with open(f"{pickle_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_mRMR, f)

        with open(f"{pickle_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_mRMR, f)
        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal)

            mRMR_indexes = load_mRMR_indexes(dataset, idx_external, idx_internal)

            X_train_mRMR = X_train[map(str, mRMR_indexes)].values
            X_test_mRMR = X_test[map(str, mRMR_indexes)].values

            pickle_file = f"{paths.MRMR_PATH}/{dataset}_mRMR_{idx_external}_{idx_internal}"

            with open(f"{pickle_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_mRMR, f)

            with open(f"{pickle_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_mRMR, f)


def main(dataset: str) -> None:
    """
        Main function
    """

    print(dataset)
    save_mRMR_indexes(dataset)
    save_mRMR(dataset)


if __name__ == '__main__':
    main(fixed_values.DATASETS[2])
