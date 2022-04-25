import itertools
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import sklearn.model_selection

from Utils import fixed_values
from Utils import paths


def load_data(dataset: str, remove_outliers: Optional[bool] = False, filter_data: Optional[bool] = False,
              easy_data: Optional[bool] = False) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Function to load data
    :param dataset: dataset identifier [CC, CDCOR, FFT]
    :param remove_outliers: remove outliers or not
    :param filter_data to get filter data
    :param easy_data to use easy data patterns only
    :return: features_names, data matrix, label vector
    """

    if remove_outliers and filter_data and easy_data:
        ValueError('Both remove_outliers, filter_data and easy_data cannot be set together.')

    if dataset == 'CC':
        data_path = paths.CC_DATA_PATH
    elif dataset == 'DCOR':
        data_path = paths.CDCOR_DATA_PATH
    elif dataset == 'FFT':
        data_path = paths.FFT_DATA_PATH
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    filter_path = '' if not filter_data else 'clean_'
    easy_path = '' if not easy_data else 'new_easy_'

    X = pd.read_pickle(f"{data_path}/{filter_path}{easy_path}X.pickle")
    y = pd.read_pickle(f"{data_path}/{filter_path}{easy_path}y.pickle")

    if remove_outliers:
        X = X.drop(fixed_values.OUTLIERS_IDX)
        y = y.drop(fixed_values.OUTLIERS_IDX)

    tt = X.columns

    if dataset == 'FFT':
        X.columns = X.columns.astype('str')

    return tt, X, y


def load_smoothed_data(idx_external: int, idx_internal: Optional[int] = None,
                       filter_data: Optional[bool] = False, easy_data: Optional[bool] = False
                       ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
        Function to load FFT data already smoothed

    :param idx_external: External fold index
    :param idx_internal: Internal fold index
    :param filter_data: If use filter_data
    :param easy_data: If use easy_data
    """
    assert isinstance(idx_internal, int) or idx_internal is None

    if filter_data + easy_data > 0:
        ValueError('Both filter_data and easy_data cannot be set together.')

    filter_set_folder = 'base'
    prefix = ''

    if filter_data:
        filter_set_folder = 'filtered'
        prefix = 'clean_'

    if easy_data:
        filter_set_folder = 'easy'
        prefix = 'new_easy_'

    folder_path = f"{paths.FFT_DATA_PATH}/smoothed_data/{filter_set_folder}"

    X_train = pd.read_pickle(f"{folder_path}/X_{idx_external}_train.pickle")
    X_test = pd.read_pickle(f"{folder_path}/X_{idx_external}_test.pickle")
    y = pd.read_pickle(f"{paths.FFT_DATA_PATH}/{prefix}y.pickle")
    y_train, y_test = y[X_train.index], y[X_test.index]

    tt = X_train.columns.astype('float64')
    if idx_internal is None:
        return tt, X_train, X_test, y_train, y_test

    internal_cv = sklearn.model_selection.StratifiedKFold(n_splits=fixed_values.INTERNAL_SPLITS, shuffle=True,
                                                          random_state=idx_external)
    index_train, index_test = next(itertools.islice(internal_cv.split(X_train, y_train), idx_internal, None), None)

    return tt, X_train.iloc[index_train], X_train.iloc[index_test], y_train.iloc[index_train], y_train.iloc[index_test]


def get_fold(X: pd.DataFrame, y: pd.Series, idx_external: int, idx_internal: Optional[int] = None,
             strategy: Optional[str] = 'kfold', outliers_remove_train: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get X_train, X_test, y_train, y_test corresponding to the fold
    :param X: Whole data matrix
    :param y: Whole label vector
    :param idx_external: idx of the fold of external division
    :param idx_internal: idx of the fold of internal division
    :param strategy: The strategy to make the fold of external division ['kfold', 'randomsplit']
    :param outliers_remove_train dataset of which we should remove outliers on train
    :return:  X_train, X_test, y_train, y_test
    """

    assert strategy in ['kfold', 'randomsplit']
    if outliers_remove_train is not None:
        assert outliers_remove_train in fixed_values.DATASETS

    if strategy == 'kfold':
        external_cv = sklearn.model_selection.StratifiedKFold(n_splits=fixed_values.EXTERNAL_SPLITS, shuffle=True,
                                                              random_state=0)
    else:
        external_cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=fixed_values.EXTERNAL_SPLITS_SHUFFLE,
                                                                     test_size=fixed_values.EXTERNAL_TEST_SIZE,
                                                                     random_state=0)

    index_train, index_test = next(itertools.islice(external_cv.split(X, y), idx_external, None), None)

    # Remove outliers of dataset from train if they are on the fold
    X_train, X_test, y_train, y_test = X.iloc[index_train], X.iloc[index_test], y.iloc[index_train], y.iloc[index_test]
    if outliers_remove_train is not None:
        outliers_idx = fixed_values.DATASET_OUTLIERS[outliers_remove_train]
        X_train = X_train.drop(outliers_idx, errors='ignore')
        y_train = y_train.drop(outliers_idx, errors='ignore')

    if idx_internal is None:
        return X_train, X_test, y_train, y_test

    if idx_internal is not None:
        # Dentro del interno la X y la y son los train del externo 9/10 del total
        X_int, y_int = X_train.copy(), y_train.copy()  # Already outliers remove if set

        # Usamos el idx externo para el random del shuffle del interno
        internal_cv = sklearn.model_selection.StratifiedKFold(n_splits=fixed_values.INTERNAL_SPLITS, shuffle=True,
                                                              random_state=idx_external)

        index_train, index_test = next(itertools.islice(internal_cv.split(X_int, y_int), idx_internal, None), None)

        return X_int.iloc[index_train], X_int.iloc[index_test], y_int.iloc[index_train], y_int.iloc[index_test]


def get_all_permutations(dictionary: Dict) -> List[Dict]:
    """
        Return all posible combinations of (key,value) from a dict

    """
    keys, values = zip(*dictionary.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def filter_patterns(fhr: pd.DataFrame, uc: pd.DataFrame, y: pd.Series, mins_cut: Optional[int] = 30,
                    nan_percentage_threshold: Optional[int] = 30) -> [pd.DataFrame, pd.DataFrame, pd.Series]:
    """

    :param fhr: FHR data
    :param uc: UC data
    :param y: labels
    :param mins_cut: mins from the end to keep
    :param nan_percentage_threshold: min nan threshold to keep a pattern
    :return: clean_fhr, clean_uc, clean_y
    """
    cut_fhr = fhr.copy().iloc[:, -mins_cut * 60 * 4:]
    fhr_nans_percent = (cut_fhr.isna().sum(axis=1) / cut_fhr.shape[1] * 100)
    fhr_now_dismissed = fhr_nans_percent.index[fhr_nans_percent > nan_percentage_threshold].to_list()

    cut_uc = uc.copy().iloc[:, -mins_cut * 60 * 4:]
    uc_nans_percent = (cut_uc.isna().sum(axis=1) / cut_fhr.shape[1] * 100)
    uc_now_dismissed = uc_nans_percent.index[uc_nans_percent > nan_percentage_threshold].to_list()

    removed = list(set(fhr_now_dismissed).union(set(uc_now_dismissed)))
    clean_fhr = cut_fhr.drop(removed)
    clean_uc = cut_uc.drop(removed)
    clean_y = y.drop(removed)

    before_dismissed = [1104, 1119, 1134, 1149, 1155, 1158, 1186, 1188, 1258, 1292, 1322,
                        1327, 1451, 1477, 1482, 2003]
    check_list = [x for x in before_dismissed if (x not in fhr_now_dismissed) and (x not in uc_now_dismissed)]

    if len(check_list) != 0:
        Warning(
            f'Not all before dismissed are removed with this parameters. '
            f'mins_cut = {mins_cut}, nan_percentage_threshold = {nan_percentage_threshold}, {check_list}'
        )

    return clean_fhr, clean_uc, clean_y


def _print_functions() -> None:
    """
        Just print defined functions docstring
    """
    print(load_data.__doc__)
    print(get_fold.__doc__)
    print(get_all_permutations.__doc__)


if __name__ == '__main__':
    _print_functions()
