"""
    Experiments to choose perfect number of njobs
"""
import time

import joblib
import numpy as np
import sklearn
from tqdm import tqdm

from Experiments.experiments import parallel_param_validation
from Preprocessing import preprocessing
from Utils import common_functions, fixed_values


def test(testing_n_jobs: int) -> None:
    n_idx_internal = 3
    strategy = 'randomsplit'

    progress_bar = tqdm(fixed_values.DATASETS,
                        total=(2 * (len(fixed_values.CLASSIFIERS) - 2) + len(fixed_values.CLASSIFIERS)) *
                              (len(fixed_values.PREPROCESSES) + 1) * n_idx_internal)

    for dataset in progress_bar:
        _, X, y = common_functions.load_data(dataset)
        for classifier_name, classifier in fixed_values.CLASSIFIERS.items():
            if dataset not in classifier['datasets']:
                continue
            clf_to_val = classifier['clf']
            for preprocess in fixed_values.PREPROCESSES + ['whole']:

                for idx_external in range(n_idx_internal):

                    tqdm_desc = f"Dataset: {dataset} " \
                                f"({(list(fixed_values.DATASETS)).index(dataset) + 1}" \
                                f"/{len(fixed_values.DATASETS)}) " \
                                f"Clf: {classifier_name} " \
                                f"({(list(fixed_values.CLASSIFIERS.keys())).index(classifier_name) + 1}" \
                                f"/{len(fixed_values.CLASSIFIERS)}) " \
                                f"Pre: {preprocess} " \
                                f"({(fixed_values.PREPROCESSES + ['whole']).index(preprocess) + 1}" \
                                f"/{len(fixed_values.PREPROCESSES) + 1}) " \
                                f"Ext. fold: {idx_external + 1}/{n_idx_internal}"
                    progress_bar.set_description(tqdm_desc)

                    if preprocess not in ['mRMR', 'whole'] or dataset != 'CC':
                        if progress_bar.last_print_n < progress_bar.total:
                            progress_bar.update(1)
                        else:
                            progress_bar.close()
                        break

                    param_grid = classifier['param_grid']
                    best_score = -1
                    best_params = -1
                    best_features_number = -1

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
                            X_train_pre, X_test_pre = preprocessing.load_preprocess(dataset, preprocess, idx_external,
                                                                                    idx_internal)

                        X_train_pre_save.append(X_train_pre)
                        X_test_pre_save.append(X_test_pre)

                    if preprocess == 'whole':
                        dimension_grid = [None]
                    else:
                        dimension_grid = fixed_values.DIMENSION_GRID

                    param_permutations = common_functions.get_all_permutations(param_grid)

                    for features_number in dimension_grid:
                        for params in param_permutations:

                            progress_bar.set_postfix(
                                {'feat': features_number, 'params': params}
                            )

                            internal_results = joblib.Parallel(n_jobs=8)(
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

                    X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, strategy=strategy)

                    if preprocess == 'whole':
                        X_train_pre_f, X_test_pre_f = X_train.copy(), X_test.copy()
                    else:
                        X_train_pre, X_test_pre = preprocessing.load_preprocess(dataset, preprocess, idx_external)
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

                    if progress_bar.last_print_n < progress_bar.total:
                        progress_bar.update(1)
                    else:
                        progress_bar.close()


def main() -> None:
    """
        Main function
    """

    for n_jobs in range(2, 9):
        t0 = time.perf_counter_ns()
        test(testing_n_jobs=n_jobs)
        t1 = time.perf_counter_ns()
        print(f"\n{n_jobs} -> {(t1 - t0) / 1e9}\n")


if __name__ == '__main__':
    main()
