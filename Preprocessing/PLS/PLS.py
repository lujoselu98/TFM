"""
    All the functions related to PLS preprocessing
"""
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import sklearn.cross_decomposition
from tqdm import tqdm

from Utils import common_functions, fixed_values, paths


def calculate_PLS(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_components: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """

    Culate the projection by PLS for Train and Test patterns

    :param X_train: Data Matrix from train
    :param y_train: Labels for train data
    :param X_test: Data Matrix from test
    :param n_components: Number of components to project into
    :return: Projection of X_train and X_test
    """
    if X_train.columns.dtype == 'float':
        X_train.columns = X_train.columns.astype('str')
        X_test.columns = X_train.columns.astype('str')

    pls_regressor = sklearn.cross_decomposition.PLSRegression(n_components=n_components)
    pls_regressor = pls_regressor.fit(X_train, y_train)

    X_train_pls = pls_regressor.transform(X_train)
    X_test_pls = pls_regressor.transform(X_test)

    return X_train_pls, X_test_pls


def save_PLS(dataset: str, strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
             filter_data: Optional[bool] = False, remove_dataset_outliers: Optional[bool] = False,
             easy_data: Optional[bool] = False) -> None:
    """
        Save the data of projected data by PLS to use in experiments
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

    tt, X, y = common_functions.load_data(dataset, remove_outliers, filter_data, easy_data)

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    remove_outliers_dataset = None
    if remove_outliers:
        save_path = paths.PLS_OUTLIERS_PATH
    elif remove_dataset_outliers:
        save_path = paths.PLS_DATASET_OUTLIERS_PATH
        remove_outliers_dataset = dataset
    else:
        save_path = paths.PLS_PATH

    filter_path = '' if not filter_data else 'clean_'
    easy_path = '' if not easy_data else 'new_easy_'

    for idx_external in tqdm(range(EXTERNAL_SPLITS)):

        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy,
                                                                     outliers_remove_train=remove_outliers_dataset)

        X_train_pls, X_test_pls = calculate_PLS(X_train, y_train, X_test, n_components=fixed_values.MAX_DIMENSION)

        components_file = f"{save_path}/{filter_path}{easy_path}{dataset}_PLS_{idx_external}"

        with open(f"{components_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_pls, f)

        with open(f"{components_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_pls, f)

        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal,
                                                                         strategy=strategy,
                                                                         outliers_remove_train=remove_outliers_dataset)

            X_train_pca, X_test_pca = calculate_PLS(X_train, y_train, X_test, n_components=fixed_values.MAX_DIMENSION)

            components_file = f"{save_path}/{filter_path}{easy_path}{dataset}_PLS_{idx_external}_{idx_internal}"

            with open(f"{components_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_pca, f)

            with open(f"{components_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_pca, f)


def main() -> None:
    """
        Main Function
    """
    for dataset in fixed_values.DATASETS:
        print(dataset)
        save_PLS(dataset, strategy='randomsplit', remove_outliers=False, filter_data=False,
                 remove_dataset_outliers=False, easy_data=True)


if __name__ == '__main__':
    main()
