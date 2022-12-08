"""
    Module for running experiments on the ECG 200 dataset.
"""
import time
from collections import Counter
from typing import Dict, List, Optional, cast

import joblib
import sklearn
from tqdm import tqdm

from Preprocessing.FPCA import FPCA
from Preprocessing.mRMR import mRMR
from Preprocessing.PLS import PLS
from Utils import common_functions, fixed_values, paths


def parallel_param_validation(
        X, y, tt,
        idx_external, idx_internal,
        clf_to_val,
        preprocess,
        features_number,
        params) -> float:
    """
    Function to parallelize the parameter validation
    :param X: Data matrix
    :param y: Labels vector
    :param tt: Indexes of the time series
    :param idx_external: Index of the external validation
    :param idx_internal: index of the internal validation
    :param clf_to_val: classifier to validate
    :param preprocess: preprocess to apply
    :param features_number: features number after preprocessing
    :param params: params for classifier to validate
    :return: metric of the model
    """

    X_train, X_test, y_train, y_test = common_functions.get_fold(X=X,
                                                                 y=y,
                                                                 idx_external=idx_external,
                                                                 idx_internal=idx_internal,
                                                                 strategy='randomsplit')
    if preprocess == 'mRMR':
        indexes = mRMR.calculate_mRMR_skfda(X_train=X_train,
                                            tt=tt,
                                            y_train=y_train,
                                            features_number=features_number)
        X_train = X_train[:, indexes]
        X_test = X_test[:, indexes]
    elif preprocess == 'PCA':
        X_train, X_test = FPCA.calculate_FPCA(X_train=X_train,
                                              X_test=X_test,
                                              tt=tt,
                                              n_components=features_number)
    elif preprocess == 'PLS':
        X_train, X_test = PLS.calculate_PLS(X_train=X_train,
                                            X_test=X_test,
                                            y_train=y_train,
                                            n_components=features_number)
    # Reset classifier
    clf = sklearn.clone(clf_to_val)

    # Set params
    clf.set_params(**params)

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    metric_test = fixed_values.VALIDATION_METRIC(y_test, y_pred)

    return metric_test


def main() -> None:
    """
        Main function for running experiments on the ECG 200 dataset.
    :return: None
    """

    progress_bar = tqdm(
        fixed_values.CLASSIFIERS,
        total=len(fixed_values.CLASSIFIERS.items()) * (
                len(fixed_values.PREPROCESSES) + 1) * fixed_values.EXTERNAL_SPLITS_SHUFFLE,
    )

    # Create file for storing results
    results_file = f"{paths.RESULTS_PATH}/results_{time.time()}_ECG_200_experiment.csv"
    with open(results_file, "a") as f:
        f.write(
            "DATASET;PREPROCESS;CLASSIFIER_NAME;"
            "IDX_EXTERNAL;FEATURES_NUMBER;PARAMS;"
            "METRICS_DICT\n"
        )

    # Load the ECG 200 dataset
    tt, X, y = common_functions.load_ucr_ECG_200_data()

    # Run experiments
    for classifier_name, classifier in fixed_values.CLASSIFIERS.items():
        clf_to_val = classifier['clf']
        for preprocess in fixed_values.PREPROCESSES + ["whole"]:
            for idx_external in range(fixed_values.EXTERNAL_SPLITS_SHUFFLE):
                tqdm_desc = (
                    f"Clf: {classifier_name} "
                    f"({(list(fixed_values.CLASSIFIERS.keys())).index(classifier_name) + 1}"
                    f"/{len(fixed_values.CLASSIFIERS)}) "
                    f"Pre: {preprocess} "
                    f"({(fixed_values.PREPROCESSES + ['whole']).index(preprocess) + 1}"
                    f"/{len(fixed_values.PREPROCESSES) + 1}) "
                    f"Ext. fold: {idx_external + 1}/{fixed_values.EXTERNAL_SPLITS_SHUFFLE}"
                )
                progress_bar.set_description(tqdm_desc)

                # Params internal validation
                param_grid = classifier['param_grid']
                param_permutations = common_functions.get_all_permutations(
                    param_grid
                )

                best_score = -1
                best_params: Dict = dict()
                best_features_number: Optional[int] = -1

                dimension_grid = cast(
                    List[Optional[int]], fixed_values.DIMENSION_GRID
                )
                if preprocess == "whole":
                    dimension_grid = [None]

                for features_number in dimension_grid:
                    for params in param_permutations:
                        progress_bar.set_postfix(
                            {"feat": features_number, "params": params}
                        )

                if progress_bar.last_print_n < progress_bar.total:
                    progress_bar.update(1)
                else:
                    progress_bar.close()


if __name__ == '__main__':
    main()
