"""
    All the functions related to mRMR preprocessing
"""
import operator
import pickle
from typing import Optional, List, Any

import dcor
import joblib
import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction import variable_selection
from tqdm import tqdm
from tqdm.notebook import tqdm as nbtqdm

from Utils import common_functions, fixed_values, paths


def calculate_mRMR_skfda(X_train: pd.DataFrame, tt: pd.Series, y_train: pd.Series, features_number: int) -> List[int]:
    """

    :param X_train: Data matrix (features as columns, patterns as rows)
    :param tt: grid points for fda data
    :param y_train: Label matrix
    :param features_number: Number of features to select
    :return: Selected features names
    """

    X_train_fd = skfda.FDataGrid(X_train, tt)
    mrmr = variable_selection.MinimumRedundancyMaximumRelevance(
        n_features_to_select=features_number,
        dependence_measure=dcor.u_distance_correlation_sqr,
        criterion=operator.sub,
    )

    return tt[mrmr.fit(X_train_fd, y_train).results_]


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


def parallel_external_code(X: pd.DataFrame, tt: pd.Series, y: pd.Series, idx_external: int,
                           strategy: Optional[str] = 'kfold',
                           remove_outliers_dataset: Optional[str] = None) -> List[int]:
    """
        Function to parallelize outer loop
    """
    assert strategy in ['kfold', 'randomsplit']
    X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy,
                                                                 outliers_remove_train=remove_outliers_dataset)

    selected_features_index = calculate_mRMR_skfda(X_train, tt, y_train,
                                                   features_number=fixed_values.MAX_DIMENSION)

    return selected_features_index


def smoothed_parallel_external_code(idx_external: int,
                                    filter_data: Optional[bool] = False,
                                    easy_data: Optional[bool] = False) -> List[int]:
    """
        Function to parallelize outer loop
    """

    tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external,
                                                                               filter_data=filter_data,
                                                                               easy_data=easy_data)

    selected_features_index = calculate_mRMR_skfda(X_train, pd.Series(tt), y_train,
                                                   features_number=fixed_values.MAX_DIMENSION)

    return selected_features_index


def parallel_internal_code(X: pd.DataFrame, tt: pd.Series, y: pd.Series, idx_external: int, idx_internal: int,
                           strategy: Optional[str] = 'kfold',
                           remove_outliers_dataset: Optional[str] = None) -> List[int]:
    """
        Function to parallelize inner loop
    """
    assert strategy in ['kfold', 'randomsplit']
    X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal, strategy=strategy,
                                                                 outliers_remove_train=remove_outliers_dataset)

    selected_features_index = calculate_mRMR_skfda(X_train, tt, y_train,
                                                   features_number=fixed_values.MAX_DIMENSION)

    return selected_features_index


def smoothed_parallel_internal_code(idx_external: int, idx_internal: int,
                                    filter_data: Optional[bool] = False,
                                    easy_data: Optional[bool] = False) -> List[int]:
    """
        Function to parallelize inner loop
    """
    tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external, idx_internal,
                                                                               filter_data=filter_data,
                                                                               easy_data=easy_data)

    selected_features_index = calculate_mRMR_skfda(X_train, pd.Series(tt), y_train,
                                                   features_number=fixed_values.MAX_DIMENSION)

    return selected_features_index


def save_mRMR_indexes(dataset: str, strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
                      filter_data: Optional[bool] = False, remove_dataset_outliers: Optional[bool] = False,
                      easy_data: Optional[bool] = False) -> None:
    """
        Save mRMR indexes into .txt files
    :param dataset: Dataset to use
    :param remove_outliers: to remove outliers or not
    :param filter_data to get filter data
    :param remove_dataset_outliers: to remove outliers or not but from dataset only
    :param easy_data to use easy data patterns only
    :param strategy: Strategy to split data
    """
    assert strategy in ['kfold', 'randomsplit']

    if remove_outliers + filter_data + remove_dataset_outliers + easy_data > 1:
        ValueError('Both remove_outliers, filter_data, remove_dataset_outliers, easy_data cannot be set together.')

    tt, X, y = common_functions.load_data(dataset, remove_outliers=remove_outliers, filter_data=filter_data,
                                          easy_data=easy_data)

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    remove_outliers_dataset = None
    if remove_dataset_outliers:
        remove_outliers_dataset = dataset

    selected_features_indexes_ext = joblib.Parallel(n_jobs=8)(
        joblib.delayed(parallel_external_code)(X, tt, y, idx_external, strategy, remove_outliers_dataset)
        for idx_external in tqdm(range(EXTERNAL_SPLITS), desc=f'mRMR {dataset}')
    )

    if remove_outliers:
        save_path = paths.MRMR_OUTLIERS_PATH
    elif remove_dataset_outliers:
        save_path = paths.MRMR_DATASET_OUTLIERS_PATH
    else:
        save_path = paths.MRMR_PATH
    filter_path = '' if not filter_data else 'clean_'
    easy_path = '' if not easy_data else 'new_easy_'

    for i, idx_external in tqdm(enumerate(range(EXTERNAL_SPLITS)), desc=f'Saving .txt {dataset}',
                                total=EXTERNAL_SPLITS):
        selected_features_index = selected_features_indexes_ext[i]

        sel_features_file = f"{save_path}/{filter_path}{easy_path}{dataset}_sel_features_{idx_external}.txt"
        with open(sel_features_file, 'w') as f:
            f.write(str(selected_features_index.tolist()))

        selected_features_indexes_int = joblib.Parallel(n_jobs=8)(
            joblib.delayed(parallel_internal_code)(X, tt, y, idx_external, idx_internal, strategy,
                                                   remove_outliers_dataset)
            for idx_internal in range(fixed_values.INTERNAL_SPLITS)
        )

        for j, idx_internal in enumerate(range(fixed_values.INTERNAL_SPLITS)):
            selected_features_index = selected_features_indexes_int[j]

            sel_features_file = f"{save_path}/{filter_path}{easy_path}{dataset}_sel_features_{idx_external}_{idx_internal}.txt"
            with open(sel_features_file, 'w') as f:
                f.write(str(selected_features_index.tolist()))


def smoothed_save_mRMR_indexes(filter_data, easy_data) -> None:
    """
        Save mRMR indexes into .txt files
    :param filter_data to get filter data
    :param easy_data to use easy data patterns only
    """
    if filter_data + easy_data > 1:
        ValueError('Both filter_data and easy_data cannot be set together.')

    selected_features_indexes_ext = joblib.Parallel(n_jobs=8)(
        joblib.delayed(smoothed_parallel_external_code)(idx_external, filter_data=filter_data, easy_data=easy_data)
        for idx_external in tqdm(range(fixed_values.EXTERNAL_SPLITS_SHUFFLE), desc=f'mRMR')
    )

    filter_set_folder = 'base'

    if filter_data:
        filter_set_folder = 'filtered'

    if easy_data:
        filter_set_folder = 'easy'

    folder_path = f"{paths.MRMR_PATH}/../smoothed/{filter_set_folder}"

    for i, idx_external in tqdm(enumerate(range(fixed_values.EXTERNAL_SPLITS_SHUFFLE)), desc=f'Saving .txt',
                                total=fixed_values.EXTERNAL_SPLITS_SHUFFLE):
        selected_features_index = selected_features_indexes_ext[i]
        sel_features_file = f"{folder_path}/sel_features_{idx_external}.txt"

        with open(sel_features_file, 'w') as f:
            f.write(str(selected_features_index.tolist()))

        selected_features_indexes_int = joblib.Parallel(n_jobs=8)(
            joblib.delayed(smoothed_parallel_internal_code)(idx_external, idx_internal,
                                                            filter_data=filter_data, easy_data=easy_data)
            for idx_internal in range(fixed_values.INTERNAL_SPLITS)
        )

        for j, idx_internal in enumerate(range(fixed_values.INTERNAL_SPLITS)):
            selected_features_index = selected_features_indexes_int[j]

            sel_features_file = f"{folder_path}/sel_features_{idx_external}_{idx_internal}.txt"
            with open(sel_features_file, 'w') as f:
                f.write(str(selected_features_index.tolist()))


def load_mRMR_indexes(dataset: str, idx_external: int, idx_internal: Optional[int] = None,
                      remove_outliers: Optional[bool] = False, filter_data: Optional[bool] = False,
                      remove_dataset_outliers: Optional[bool] = False, easy_data: Optional[bool] = False) -> List[int]:
    """
        Load indexes from .txt file, used to save the real values needed for experiments
    :param dataset: Dataset to load
    :param remove_outliers: to remove outliers or not
    :param filter_data to get filter data
    :param remove_dataset_outliers: to remove outliers or not but from dataset only
    :param easy_data to use easy data patterns only
    :param idx_external: External idx to load
    :param idx_internal: Internal idx to load
    :return: Read indexes
    """

    if remove_outliers + filter_data + remove_dataset_outliers + easy_data > 1:
        ValueError('Both remove_outliers, filter_data, remove_dataset_outliers, easy_data cannot be set together.')

    filter_path = '' if not filter_data else 'clean_'
    easy_path = '' if not easy_data else 'new_easy_'

    if remove_outliers:
        load_path = paths.MRMR_OUTLIERS_PATH
    elif remove_dataset_outliers:
        load_path = paths.MRMR_DATASET_OUTLIERS_PATH
    else:
        load_path = paths.MRMR_PATH

    if idx_internal is None:
        mRMR_indexes_file = f"{load_path}/{filter_path}{easy_path}{dataset}_sel_features_{idx_external}.txt"

    else:  # idx_internal is not None
        mRMR_indexes_file = f"{load_path}/{filter_path}{easy_path}{dataset}_sel_features_{idx_external}_{idx_internal}.txt"

    with open(f'{mRMR_indexes_file}', 'r') as f:
        line = f.readline()
        mRMR_indexes: List[Any] = line.strip().replace("]", "").replace("[", "").replace("'", "").split(', ')
        if dataset != 'FFT':
            mRMR_indexes = [int(x) for x in mRMR_indexes]
        else:
            mRMR_indexes = [float(x) for x in mRMR_indexes]
    return mRMR_indexes


def smoothed_load_mRMR_indexes(idx_external: int, idx_internal: Optional[int] = None,
                               filter_data: Optional[bool] = False, easy_data: Optional[bool] = False):
    """
        Load indexes from .txt file, used to save the real values needed for experiments
    :param filter_data to get filter data
    :param easy_data to use easy data patterns only
    :param idx_external: External idx to load
    :param idx_internal: Internal idx to load
    :return: Read indexes
    """

    if filter_data + easy_data > 1:
        ValueError('Both filter_data and easy_data cannot be set together.')

    filter_set_folder = 'base'

    if filter_data:
        filter_set_folder = 'filtered'

    if easy_data:
        filter_set_folder = 'easy'

    folder_path = f"{paths.MRMR_PATH}/../smoothed/{filter_set_folder}"

    if idx_internal is None:
        mRMR_indexes_file = f"{folder_path}/sel_features_{idx_external}.txt"
    else:
        mRMR_indexes_file = f"{folder_path}/sel_features_{idx_external}_{idx_internal}.txt"

    with open(f'{mRMR_indexes_file}', 'r') as f:
        line = f.readline()
        mRMR_indexes: List[Any] = line.strip().replace("]", "").replace("[", "").replace("'", "").split(', ')
        mRMR_indexes = [float(x) for x in mRMR_indexes]

    return mRMR_indexes


def save_mRMR(dataset: str, strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
              filter_data: Optional[bool] = False, remove_dataset_outliers: Optional[bool] = False,
              easy_data: Optional[bool] = False) -> None:
    """
        Load indexes from .txt file, used to save the real values needed for experiments
    :param dataset: Dataset to load
    :param remove_outliers: to remove outliers or not
    :param filter_data to get filter data
    :param remove_dataset_outliers: to remove outliers or not but from dataset only
    :param easy_data to use easy data patterns only
    :param strategy: Strategy to split data
    """
    assert strategy in ['kfold', 'randomsplit']

    if remove_outliers + filter_data + remove_dataset_outliers + easy_data > 1:
        ValueError('Both remove_outliers, filter_data, remove_dataset_outliers, easy_data cannot be set together.')

    tt, X, y = common_functions.load_data(dataset, remove_outliers=remove_outliers, filter_data=filter_data,
                                          easy_data=easy_data)

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    if remove_outliers:
        save_path = paths.MRMR_OUTLIERS_PATH
    elif remove_dataset_outliers:
        save_path = paths.MRMR_DATASET_OUTLIERS_PATH
    else:
        save_path = paths.MRMR_PATH

    filter_path = '' if not filter_data else 'clean_'
    easy_path = '' if not easy_data else 'new_easy_'

    for idx_external in tqdm(range(EXTERNAL_SPLITS), desc=f'Saving .pickle {dataset}'):
        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy)
        mRMR_indexes: List[Any] = load_mRMR_indexes(dataset, idx_external, remove_outliers=remove_outliers,
                                                    remove_dataset_outliers=remove_dataset_outliers)

        if dataset == 'FFT':
            mRMR_indexes = list(map(str, mRMR_indexes))

        X_train_mRMR = X_train[mRMR_indexes].values
        X_test_mRMR = X_test[mRMR_indexes].values

        pickle_file = f"{save_path}/{filter_path}{easy_path}{dataset}_mRMR_{idx_external}"

        with open(f"{pickle_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_mRMR, f)

        with open(f"{pickle_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_mRMR, f)
        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal,
                                                                         strategy=strategy)

            mRMR_indexes = load_mRMR_indexes(dataset, idx_external, idx_internal, remove_outliers=remove_outliers)

            if dataset == 'FFT':
                mRMR_indexes = list(map(str, mRMR_indexes))

            X_train_mRMR = X_train[mRMR_indexes].values
            X_test_mRMR = X_test[mRMR_indexes].values

            pickle_file = f"{save_path}/{filter_path}{easy_path}{dataset}_mRMR_{idx_external}_{idx_internal}"

            with open(f"{pickle_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_mRMR, f)

            with open(f"{pickle_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_mRMR, f)


def smoothed_save_mRMR(filter_data: Optional[bool] = False, easy_data: Optional[bool] = False):
    """
       Load indexes from .txt file, used to save the real values needed for experiments
   :param filter_data to get filter data
   :param easy_data to use easy data patterns only
   """
    if filter_data + easy_data > 1:
        ValueError('Both filter_data and easy_data cannot be set together.')

    filter_set_folder = 'base'

    if filter_data:
        filter_set_folder = 'filtered'

    if easy_data:
        filter_set_folder = 'easy'

    folder_path = f"{paths.MRMR_PATH}/../smoothed/{filter_set_folder}"

    for idx_external in tqdm(range(fixed_values.EXTERNAL_SPLITS_SHUFFLE), desc=f'Saving .pickle'):
        tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external,
                                                                                   filter_data=filter_data,
                                                                                   easy_data=easy_data)
        mRMR_indexes: List[Any] = smoothed_load_mRMR_indexes(idx_external,
                                                             filter_data=filter_data,
                                                             easy_data=easy_data)
        mRMR_indexes = list(map(str, mRMR_indexes))
        X_train_mRMR = X_train[mRMR_indexes].values
        X_test_mRMR = X_test[mRMR_indexes].values

        pickle_file = f"{folder_path}/mRMR_{idx_external}"

        with open(f"{pickle_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_mRMR, f)

        with open(f"{pickle_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_mRMR, f)
        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external, idx_internal,
                                                                                       filter_data=filter_data,
                                                                                       easy_data=easy_data)

            mRMR_indexes = smoothed_load_mRMR_indexes(idx_external, idx_internal,
                                                      filter_data=filter_data,
                                                      easy_data=easy_data)
            mRMR_indexes = list(map(str, mRMR_indexes))

            X_train_mRMR = X_train[mRMR_indexes].values
            X_test_mRMR = X_test[mRMR_indexes].values

            pickle_file = f"{folder_path}/mRMR_{idx_external}_{idx_internal}"

            with open(f"{pickle_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_mRMR, f)

            with open(f"{pickle_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_mRMR, f)


def main(dataset: str, filter_set: str) -> None:
    """
        Main function
    """

    # print(dataset)
    # save_mRMR_indexes(dataset,
    #                   strategy='randomsplit',
    #                   remove_outliers=False, filter_data=False, remove_dataset_outliers=False, easy_data=True)
    # save_mRMR(dataset,
    #           strategy='randomsplit',
    #           remove_outliers=False, filter_data=False, remove_dataset_outliers=False, easy_data=True)
    # smoothed_save_mRMR_indexes(filter_data=filter_set == 'filtered', easy_data= filter_set == 'easy')
    smoothed_save_mRMR(filter_data=filter_set == 'filtered', easy_data= filter_set == 'easy')

if __name__ == '__main__':
    main(fixed_values.DATASETS[1], filter_set='Base')
