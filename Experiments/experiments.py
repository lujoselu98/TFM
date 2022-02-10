"""
    Main File to do Different Experiments and save the results
"""
import time

import joblib
import numpy as np
import sklearn
from tqdm import tqdm

from Preprocessing import preprocessing
from Utils import fixed_values, paths, common_functions


def main_experiment() -> None:
    """Function to made the main experiment"""

    progress_bar = tqdm(fixed_values.DATASETS,
                        total=(2 * (len(fixed_values.CLASSIFIERS) - 2) + len(fixed_values.CLASSIFIERS)) *
                              (len(fixed_values.PREPROCESSES) + 1) * fixed_values.EXTERNAL_SPLITS)

    results_file = f"{paths.RESULTS_PATH}/results_{time.time()}_main_experiment.csv"
    with open(results_file, 'a') as f:
        f.write("DATASET;PREPROCESSES;CLASSIFIER_NAME;"
                "IDX_EXTERNAL;FEATURES_NUMBER;PARAMS;"
                "METRICS_DICT")
    for dataset in progress_bar:
        _, X, y = common_functions.load_data(dataset)
        for classifier_name, classifier in fixed_values.CLASSIFIERS.items():
            if dataset not in classifier['datasets']:
                continue
            clf_to_val = classifier['clf']
            for preprocess in ['whole'] + fixed_values.PREPROCESSES:
                for idx_external in range(fixed_values.EXTERNAL_SPLITS):
                    tqdm_desc = f"Dataset {dataset} " \
                                f"Clf: {classifier_name} " \
                                f"({(list(fixed_values.CLASSIFIERS.keys())).index(classifier_name) + 1}" \
                                f"/{len(fixed_values.CLASSIFIERS)}) " \
                                f"Pre: {preprocess} " \
                                f"({(['whole'] + fixed_values.PREPROCESSES).index(preprocess) + 1}" \
                                f"/{len(fixed_values.PREPROCESSES) + 1}) " \
                                f"Ext. fold: {idx_external + 1}/{fixed_values.EXTERNAL_SPLITS}"
                    progress_bar.set_description(tqdm_desc)

                    # Params internal validation
                    param_grid = classifier['param_grid']
                    best_score = -1
                    best_params = -1
                    best_features_number = -1

                    param_permutatios = common_functions.get_all_permutations(param_grid)

                    y_train_save, y_test_save = [], []
                    X_train_pre_save, X_test_pre_save = [], []
                    for idx_internal in range(fixed_values.INTERNAL_SPLITS):
                        X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external, idx_internal)
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

                    for features_number in dimension_grid:
                        for params in param_permutatios:
                            internal_results = []
                            for idx_internal in range(fixed_values.INTERNAL_SPLITS):
                                progress_bar.set_postfix(
                                    {'feat': features_number, 'params': params, 'idx_int': idx_internal}
                                )
                                y_train, y_test = y_train_save[idx_internal], y_test_save[idx_internal]
                                X_train_pre, X_test_pre = X_train_pre_save[idx_internal], X_test_pre_save[idx_internal]

                                if preprocess == 'whole':
                                    X_train_pre_f, X_test_pre_f = X_train_pre.copy(), X_test_pre.copy()
                                else:
                                    X_train_pre_f, X_test_pre_f = preprocessing.get_features(X_train_pre, X_test_pre,
                                                                                             features_number)
                                # Reset
                                clf = sklearn.clone(clf_to_val)

                                # Set params
                                clf.set_params(**params)

                                # Train
                                clf.fit(X_test_pre_f, y_train)

                                # Evaluate
                                y_pred = clf.predict(X_test_pre_f)
                                metric_test = fixed_values.VALIDATION_METRIC(y_test, y_pred)

                                # Save results
                                internal_results.append(metric_test)

                            # Compare and take best (model, params, features)
                            mean_score = np.mean(internal_results)

                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                best_features_number = features_number

                    # End of param cross validation

                    X_train, X_test, y_train, y_test = common_functions.get_fold(X, y, idx_external)

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
                    clf.fit(X_train_pre_f, X_test_pre_f)
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
                    joblib.dump(clf,
                                f'{paths.CLASSIFIERS_PATH}/{dataset}_{preprocess}_{classifier_name}'
                                f'_{idx_external}.joblib')

                    csv_string = f"{dataset};{preprocess};{classifier_name};{idx_external};" \
                                 f"{best_features_number};{best_params};{metrics_dict}"
                    with open(f'{results_file}', 'a') as f:
                        f.write(f"{csv_string}\n")

                    if progress_bar.last_print_n < progress_bar.total:
                        progress_bar.update(1)
                    else:
                        progress_bar.close()


if __name__ == '__main__':
    main_experiment()
