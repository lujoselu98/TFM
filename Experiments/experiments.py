"""
    Main File to do Different Experiments and save the results
"""
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

from Preprocessing import preprocessing
from Utils import fixed_values, paths, common_functions


def parallel_param_validation(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series,
                              clf_to_val: sklearn.base.ClassifierMixin, preprocess: str,
                              features_number: int, params: dict) -> float:
    """

    Parallel validation

    """

    if preprocess == 'whole':
        X_train_pre_f, X_test_pre_f = X_train.copy(), X_test.copy()
    else:
        X_train_pre_f, X_test_pre_f = preprocessing.get_features(X_train, X_test, features_number)

    # Reset
    clf = sklearn.clone(clf_to_val)

    # Set params
    clf.set_params(**params)

    # Train
    clf.fit(X_train_pre_f, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_pre_f)
    metric_test = fixed_values.VALIDATION_METRIC(y_test, y_pred)

    # Save results
    return metric_test


def main_experiment(strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
                    filter_data: Optional[bool] = False, easy_data: Optional[bool] = False) -> None:
    """Function to made the main experiment"""

    assert strategy in ['kfold', 'randomsplit']

    if remove_outliers + filter_data + easy_data > 1:
        ValueError('Both remove_outliers, filter_data and easy_data cannot be set together.')

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    progress_bar = tqdm(fixed_values.DATASETS,
                        total=(2 * (len(fixed_values.CLASSIFIERS) - 1) + len(fixed_values.CLASSIFIERS)) *
                              (len(fixed_values.PREPROCESSES)) * EXTERNAL_SPLITS)
                              # (len(fixed_values.PREPROCESSES) + 1) * EXTERNAL_SPLITS)

    results_file = f"{paths.RESULTS_PATH}/results_{time.time()}_main_experiment.csv"
    with open(results_file, 'a') as f:
        f.write("DATASET;PREPROCESS;CLASSIFIER_NAME;"
                "IDX_EXTERNAL;FEATURES_NUMBER;PARAMS;"
                "METRICS_DICT\n")
    for dataset in progress_bar:
        _, X, y = common_functions.load_data(dataset, remove_outliers=remove_outliers, filter_data=filter_data,
                                             easy_data=easy_data)
        for classifier_name, classifier in fixed_values.CLASSIFIERS.items():
            if dataset not in classifier['datasets']:
                continue
            clf_to_val = classifier['clf']
            for preprocess in fixed_values.PREPROCESSES + ['whole']:
                if preprocess == 'whole':
                    continue
                for idx_external in range(EXTERNAL_SPLITS):

                    tqdm_desc = f"Dataset: {dataset} " \
                                f"({(list(fixed_values.DATASETS)).index(dataset) + 1}" \
                                f"/{len(fixed_values.DATASETS)}) " \
                                f"Clf: {classifier_name} " \
                                f"({(list(fixed_values.CLASSIFIERS.keys())).index(classifier_name) + 1}" \
                                f"/{len(fixed_values.CLASSIFIERS)}) " \
                                f"Pre: {preprocess} " \
                                f"({(fixed_values.PREPROCESSES + ['whole']).index(preprocess) + 1}" \
                                f"/{len(fixed_values.PREPROCESSES) + 1}) " \
                                f"Ext. fold: {idx_external + 1}/{EXTERNAL_SPLITS}"
                    progress_bar.set_description(tqdm_desc)

                    # Params internal validation
                    param_grid = classifier['param_grid']
                    best_score = -1
                    best_params = -1
                    best_features_number: Optional[int] = -1

                    y_train_save, y_test_save = [], []
                    X_train_pre_save, X_test_pre_save = [], []
                    for idx_internal in range(fixed_values.INTERNAL_SPLITS):
                        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal,
                                                                                     strategy=strategy)
                        y_train_save.append(y_train)
                        y_test_save.append(y_test)

                        if preprocess == 'whole':
                            X_train_pre, X_test_pre = X_train.copy(), X_test.copy()
                        else:
                            X_train_pre, X_test_pre = preprocessing.load_preprocess(dataset, preprocess,
                                                                                    idx_external, idx_internal,
                                                                                    remove_outliers=remove_outliers,
                                                                                    filter_data=filter_data,
                                                                                    easy_data=easy_data)

                        X_train_pre_save.append(X_train_pre)
                        X_test_pre_save.append(X_test_pre)

                    if preprocess == 'whole':
                        dimension_grid = [None]
                    else:
                        dimension_grid = fixed_values.DIMENSION_GRID

                    param_permutations = common_functions.get_all_permutations(param_grid)

                    for features_number in dimension_grid:
                        for params in param_permutations:

                            # progress_bar.set_postfix(
                            #     {'feat': features_number, 'params': params}
                            # )

                            internal_results = joblib.Parallel(n_jobs=5)(
                                joblib.delayed(parallel_param_validation)(
                                    X_train_pre_save[idx_internal], X_test_pre_save[idx_internal],
                                    y_train_save[idx_internal], y_test_save[idx_internal],
                                    clf_to_val, preprocess,
                                    features_number, params
                                )
                                for idx_internal in range(fixed_values.INTERNAL_SPLITS)
                            )

                            # Compare and take best (model, params, features)
                            mean_score = np.mean(internal_results)

                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                best_features_number = features_number
                    # End of param cross validation

                    X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy)

                    if preprocess == 'whole':
                        X_train_pre_f, X_test_pre_f = X_train.copy(), X_test.copy()
                    else:
                        X_train_pre, X_test_pre = preprocessing.load_preprocess(dataset, preprocess, idx_external,
                                                                                remove_outliers=remove_outliers,
                                                                                filter_data=filter_data,
                                                                                easy_data=easy_data)
                        X_train_pre_f, X_test_pre_f = preprocessing.get_features(X_train_pre, X_test_pre,
                                                                                 best_features_number)

                    # Reset
                    clf = sklearn.clone(clf_to_val)

                    # Set best params
                    clf.set_params(**best_params)

                    # Train
                    clf.fit(X_train_pre_f, y_train)
                    y_pred = clf.predict(X_test_pre_f)

                    # Evaluate and save
                    if classifier['evaluate_score'] == 'decision_function':
                        y_score = clf.decision_function(X_test_pre_f)

                    if classifier['evaluate_score'] == 'predict_proba':
                        y_score = clf.predict_proba(X_test_pre_f)[:, 1]

                    metrics_dict = {}
                    for ev_name, ev_metric in fixed_values.EVALUATION_METRICS.items():
                        if ev_metric['values'] == 'scores':
                            metrics_dict[ev_name] = ev_metric['function'](y_test, y_score)

                        if ev_metric['values'] == 'predictions':
                            metrics_dict[ev_name] = ev_metric['function'](y_test, y_pred)
                    # joblib.dump(clf,
                    #             f'{paths.CLASSIFIERS_PATH}/{dataset}_{preprocess}_{classifier_name}'
                    #             f'_{idx_external}.joblib')

                    csv_string = f"{dataset};{preprocess};{classifier_name};{idx_external};" \
                                 f"{best_features_number};{best_params};{metrics_dict}"
                    with open(f'{results_file}', 'a') as f:
                        f.write(f"{csv_string}\n")

                    if progress_bar.last_print_n < progress_bar.total:
                        progress_bar.update(1)
                    else:
                        progress_bar.close()


def dummy_classifier(strategy: Optional[str] = 'kfold', remove_outliers: Optional[bool] = False,
                     filter_data: Optional[bool] = False, easy_data: Optional[bool] = False) -> None:
    """Function to made the dummy experiment"""

    assert strategy in ['kfold', 'randomsplit']

    if remove_outliers + filter_data + easy_data > 1:
        ValueError('Both remove_outliers, filter_data and easy_data cannot be set together.')

    if strategy == 'kfold':
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS
    else:
        EXTERNAL_SPLITS = fixed_values.EXTERNAL_SPLITS_SHUFFLE

    progress_bar = tqdm(fixed_values.DATASETS,
                        total=len(fixed_values.DATASETS) * EXTERNAL_SPLITS
                        )

    results_file = f"{paths.RESULTS_PATH}/results_{time.time()}_dummy.csv"
    with open(results_file, 'a') as f:
        f.write("DATASET;PREPROCESS;CLASSIFIER_NAME;"
                "IDX_EXTERNAL;FEATURES_NUMBER;PARAMS;"
                "METRICS_DICT\n")
    classifier = fixed_values.DUMMY_CLASSIFIER
    classifier_name = 'DUMMY_CLASSIFIER'
    for dataset in progress_bar:
        _, X, y = common_functions.load_data(dataset, remove_outliers=remove_outliers, filter_data=filter_data,
                                             easy_data=easy_data)
        if dataset not in classifier['datasets']:
            continue

        base_clf = classifier['clf']
        for idx_external in range(EXTERNAL_SPLITS):
            tqdm_desc = f"Dataset: {dataset} " \
                        f"({(list(fixed_values.DATASETS)).index(dataset) + 1}" \
                        f"/{len(fixed_values.DATASETS)}) " \
                        f"Ext. fold: {idx_external + 1}/{EXTERNAL_SPLITS}"

            progress_bar.set_description(tqdm_desc)
            X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy)

            # Reset
            clf = sklearn.clone(base_clf)

            # Train
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Evaluate and save
            if classifier['evaluate_score'] == 'decision_function':
                y_score = clf.decision_function(X_test)

            if classifier['evaluate_score'] == 'predict_proba':
                y_score = clf.predict_proba(X_test)[:, 1]

            metrics_dict = {}
            for ev_name, ev_metric in fixed_values.EVALUATION_METRICS.items():
                if ev_metric['values'] == 'scores':
                    metrics_dict[ev_name] = ev_metric['function'](y_test, y_score)

                if ev_metric['values'] == 'predictions':
                    metrics_dict[ev_name] = ev_metric['function'](y_test, y_pred)

            csv_string = f"{dataset};{None};{classifier_name};{idx_external};" \
                         f"{None};{None};{metrics_dict}"

            with open(f'{results_file}', 'a') as f:
                f.write(f"{csv_string}\n")

            if progress_bar.last_print_n < progress_bar.total:
                progress_bar.update(1)
            else:
                progress_bar.close()


if __name__ == '__main__':
    # dummy_classifier(strategy='randomsplit', remove_outliers=False, filter_data=False, easy_data=True)
    main_experiment(strategy='randomsplit', remove_outliers=False, filter_data=False, easy_data=True)
