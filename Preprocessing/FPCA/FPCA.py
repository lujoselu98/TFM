"""
    All the functions related to FPCA preprocessing
"""
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction.projection import FPCA
from tqdm import tqdm

from Utils import paths, fixed_values, common_functions


def calculate_FPCA(X_train: pd.DataFrame, X_test: pd.DataFrame, tt: np.ndarray, n_components: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
        Calculate FPCA and return the transform of X_train and X_test

    :param X_train: Data matrix train patterns
    :param X_test: Data matrix test pattern
    :param tt: Grid points
    :param n_components: Number of components to calculate FPCA
    :return: X_train and X_test projection
    """
    fpca = FPCA(n_components=n_components)
    X_train_pca = fpca.fit_transform(skfda.FDataGrid(X_train, tt))
    X_test_pca = fpca.transform(skfda.FDataGrid(X_test, tt))

    return X_train_pca, X_test_pca


def save_FPCA(dataset: str, strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
              filter_data: Optional[bool] = False, remove_dataset_outliers: Optional[bool] = False,
              easy_data: Optional[bool] = False) -> None:
    """
        Save the data of projected data by FPCA to use in experiments
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
        save_path = paths.FPCA_OUTLIERS_PATH
    elif remove_dataset_outliers:
        save_path = paths.FPCA_DATASET_OUTLIERS_PATH
        remove_outliers_dataset = dataset
    else:
        save_path = paths.FPCA_PATH

    filter_path = '' if not filter_data else 'clean_'
    # easy_path = '' if not easy_data else 'new_easy_'
    easy_path = '' if not easy_data else '705_easy_'

    for idx_external in tqdm(range(EXTERNAL_SPLITS)):

        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy,
                                                                     outliers_remove_train=remove_outliers_dataset)

        X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

        components_file = f"{save_path}/{filter_path}{easy_path}{dataset}_PCA_{idx_external}"

        with open(f"{components_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_pca, f)

        with open(f"{components_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_pca, f)

        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal,
                                                                         strategy=strategy,
                                                                         outliers_remove_train=remove_outliers_dataset)

            X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

            components_file = f"{save_path}/{filter_path}{easy_path}{dataset}_PCA_{idx_external}_{idx_internal}"

            with open(f"{components_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_pca, f)

            with open(f"{components_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_pca, f)


def smoothed_save_FPCA(filter_data: Optional[bool] = False, easy_data: Optional[bool] = False) -> None:
    """
        Save the data of projected data by FPCA to use in experiments
    :param filter_data to get filter data
    :param easy_data to use easy data patterns only
    """

    if filter_data + easy_data > 1:
        ValueError('Both filter_data, easy_data cannot be set together.')

    filter_set_folder = 'base'

    if filter_data:
        filter_set_folder = 'filtered'

    if easy_data:
        filter_set_folder = 'easy'

    folder_path = f"{paths.FPCA_PATH}/../smoothed/{filter_set_folder}"

    for idx_external in tqdm(range(fixed_values.EXTERNAL_SPLITS_SHUFFLE)):
        tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external,
                                                                                   filter_data=filter_data,
                                                                                   easy_data=easy_data)

        X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

        components_file = f"{folder_path}/PCA_{idx_external}"

        with open(f"{components_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_pca, f)

        with open(f"{components_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_pca, f)

        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            tt, X_train, X_test, y_train, y_test = common_functions.load_smoothed_data(idx_external, idx_internal,
                                                                                       filter_data, easy_data)

            X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

            components_file = f"{folder_path}/PCA_{idx_external}_{idx_internal}"

            with open(f"{components_file}_train.pickle", 'wb') as f:
                pickle.dump(X_train_pca, f)

            with open(f"{components_file}_test.pickle", 'wb') as f:
                pickle.dump(X_test_pca, f)


def main() -> None:
    """
        Main Function
    """
    for dataset in fixed_values.DATASETS:
        # if dataset != 'FFT':
        #      continue
        print(dataset)
        save_FPCA(dataset, strategy='randomsplit', remove_outliers=False, filter_data=False,
                  remove_dataset_outliers=False, easy_data=True)
    # smoothed_save_FPCA(filter_data=False, easy_data=False)
    # smoothed_save_FPCA(filter_data=True, easy_data=False)
    # smoothed_save_FPCA(filter_data=False, easy_data=True)


if __name__ == '__main__':
    main()
