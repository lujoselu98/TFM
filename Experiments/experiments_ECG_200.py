"""
    Module for running experiments on the ECG 200 dataset.
"""
import time
from collections import Counter
from typing import Dict, List, Optional, cast

import joblib
import numpy as np
import sklearn
from tqdm import tqdm

from Preprocessing.FPCA import FPCA
from Preprocessing.mRMR import mRMR
from Preprocessing.PLS import PLS
from Utils import common_functions, fixed_values, paths


def parallel_param_validation(
        X_train, X_test,
        y_train, y_test,
        features_number,
        clf_to_val,
        params
) -> float:
    """
    Function to parallelize the parameter validation
    :param X_train: Train data matrix
    :param X_test: Test data matrix
    :param y_train: Train labels vector
    :param y_test: Test labels vector
    :param features_number: Number of features to use
    :param clf_to_val: classifier to validate
    :param params: params for classifier to validate
    :return: metric of the model
    """

    X_train = X_train.iloc[:, :features_number]
    X_test = X_test.iloc[:, :features_number]

    # Reset classifier
    clf = sklearn.clone(clf_to_val)

    # Set params
    clf.set_params(**params)

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    test_scores = fixed_values.VALIDATION_METRIC(y_test, y_pred)

    return test_scores


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
            "PREPROCESS;CLASSIFIER_NAME;"
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

                y_train_save, y_test_save = [], []
                X_train_save, X_test_save = [], []
                for idx_internal in range(fixed_values.INTERNAL_SPLITS):
                    X_train, X_test, y_train, y_test = common_functions.get_fold(
                        X, y, idx_external, idx_internal, strategy='randomsplit'
                    )
                    y_train_save.append(y_train)
                    y_test_save.append(y_test)

                    if preprocess == 'mRMR':
                        indexes = mRMR.calculate_mRMR_skfda(X_train=X_train,
                                                            tt=tt,
                                                            y_train=y_train,
                                                            features_number=fixed_values.MAX_DIMENSION)
                        X_train = X_train[indexes]
                        X_test = X_test[indexes]
                    elif preprocess == 'PCA':
                        X_train, X_test = FPCA.calculate_FPCA(X_train=X_train,
                                                              X_test=X_test,
                                                              tt=tt,
                                                              n_components=fixed_values.MAX_DIMENSION)
                    elif preprocess == 'PLS':
                        X_train, X_test = PLS.calculate_PLS(X_train=X_train,
                                                            X_test=X_test,
                                                            y_train=y_train,
                                                            n_components=fixed_values.MAX_DIMENSION)

                    X_train_save.append(X_train)
                    X_test_save.append(X_test)

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

                        # Internal validation
                        internal_scores = joblib.Parallel(n_jobs=5)(
                            joblib.delayed(parallel_param_validation)(
                                X_train=X_train_save[idx_internal],
                                X_test=X_test_save[idx_internal],
                                y_train=y_train_save[idx_internal],
                                y_test=y_test_save[idx_internal],
                                clf_to_val=clf_to_val,
                                features_number=features_number,
                                params=params,
                            )
                            for idx_internal in range(
                                fixed_values.INTERNAL_SPLITS
                            )
                        )
                        # internal_scores = [
                        #     parallel_param_validation(
                        #         X_train=X_train_save[idx_internal],
                        #         X_test=X_test_save[idx_internal],
                        #         y_train=y_train_save[idx_internal],
                        #         y_test=y_test_save[idx_internal],
                        #         clf_to_val=clf_to_val,
                        #         features_number=features_number,
                        #         params=params,
                        #     )
                        #     for idx_internal in range(
                        #         fixed_values.INTERNAL_SPLITS
                        #     )
                        # ]
                        mean_score = np.mean(internal_scores)

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
                            best_features_number = features_number

                progress_bar.set_postfix(
                    {"best_score": best_score,
                     "best_features_number": best_features_number,
                     "best_params": best_params}
                )

                X_train, X_test, y_train, y_test = common_functions.get_fold(
                    X=X, y=y, idx_external=idx_external, strategy='randomsplit'
                )

                if preprocess == 'mRMR':
                    indexes = mRMR.calculate_mRMR_skfda(X_train=X_train,
                                                        tt=tt,
                                                        y_train=y_train,
                                                        features_number=cast(int, best_features_number)
                                                        )
                    X_train = X_train[indexes]
                    X_test = X_test[indexes]
                elif preprocess == 'PCA':
                    X_train, X_test = FPCA.calculate_FPCA(X_train=X_train,
                                                          X_test=X_test,
                                                          tt=tt,
                                                          n_components=cast(int, best_features_number)
                                                          )
                elif preprocess == 'PLS':
                    X_train, X_test = PLS.calculate_PLS(X_train=X_train,
                                                        X_test=X_test,
                                                        y_train=y_train,
                                                        n_components=cast(int, best_features_number)
                                                        )

                # Reset
                clf = sklearn.clone(clf_to_val)

                # Set best params
                clf.set_params(**best_params)

                # Train
                clf.fit(X_train, y_train)

                # Predict
                y_pred = clf.predict(X_test)

                # Evaluate and save
                if classifier["evaluate_score"] == "decision_function":
                    y_score = clf.decision_function(X_test)

                if classifier["evaluate_score"] == "predict_proba":
                    y_score = clf.predict_proba(X_test)[:, 1]

                metrics_dict = {}
                for ev_name, ev_metric in fixed_values.EVALUATION_METRICS.items():
                    if ev_metric["values"] == "scores":
                        metrics_dict[ev_name] = ev_metric["function"](
                            y_test, y_score
                        )

                    if ev_metric["values"] == "predictions":
                        metrics_dict[ev_name] = ev_metric["function"](
                            y_test, y_pred
                        )

                csv_string = (
                    f"{preprocess};{classifier_name};{idx_external};"
                    f"{best_features_number};{best_params};{metrics_dict}"
                )

                with open(f"{results_file}", "a") as f:
                    f.write(f"{csv_string}\n")

                if progress_bar.last_print_n < progress_bar.total:
                    progress_bar.update(1)
                else:
                    progress_bar.close()


if __name__ == '__main__':
    main()
