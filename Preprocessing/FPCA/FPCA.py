"""
    All the functions related to FPCA preprocessing
"""
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction.projection import FPCA
from tqdm import tqdm

from Utils import paths, fixed_values, common_functions


def calculate_FPCA(X_train: pd.DataFrame, X_test: pd.DataFrame, tt: pd.Series, n_components: int) -> Tuple[
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


def save_PCA(dataset: str) -> None:
    """
        Save the data of projected data by FPCA to use in experiments
    :param dataset: Dataset to use
    """
    tt, X, y = common_functions.load_data(dataset)
    for idx_external in tqdm(range(fixed_values.EXTERNAL_SPLITS)):

        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external)

        X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

        components_file = f"{paths.FPCA_PATH}/{dataset}_PCA_{idx_external}"

        with open(f"{components_file}_train.pickle", 'wb') as f:
            pickle.dump(X_train_pca, f)

        with open(f"{components_file}_test.pickle", 'wb') as f:
            pickle.dump(X_test_pca, f)

        for idx_internal in range(fixed_values.INTERNAL_SPLITS):
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal)

            X_train_pca, X_test_pca = calculate_FPCA(X_train, X_test, tt, n_components=fixed_values.MAX_DIMENSION)

            components_file = f"{paths.FPCA_PATH}/{dataset}_PCA_{idx_external}_{idx_internal}"

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
        save_PCA(dataset)


if __name__ == '__main__':
    main()
