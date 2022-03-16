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


def save_FPCA(dataset: str, strategy: Optional[str] = 'kfold', remove_outliers: bool = False,
              filter_data: Optional[bool] = False) -> None:
    """
        Save the data of projected data by FPCA to use in experiments
    :param dataset: Dataset to use
    :param remove_outliers: to remove outliers or not
    :param filter_data to get filter data
    :param strategy: Strategy to split data
    """
    assert strategy in ['kfold', 'randomsplit']

    if remove_outliers and filter_data:
        ValueError('Both remove_outliers and filter_data cannot be set together.')

    tt, X, y = common_functions.load_data(dataset, remove_outliers, filter_data)

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    save_path = paths.FPCA_PATH
    if remove_outliers:
        save_path = paths.FPCA_OUTLIERS_PATH

    filter_path = '' if not filter_data else 'clean_'

    for idx_external in tqdm(range(EXTERNAL_SPLITS)):

        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy)

        X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

        components_file = f"{save_path}/{filter_path}{dataset}_PCA_{idx_external}"

        with open(f"{components_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_pca, f)

        with open(f"{components_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_pca, f)

        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal,
                                                                         strategy=strategy)

            X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

            components_file = f"{save_path}/{filter_path}{dataset}_PCA_{idx_external}_{idx_internal}"

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
        save_FPCA(dataset, strategy='randomsplit', remove_outliers=False, filter_data=True)


if __name__ == '__main__':
    main()
