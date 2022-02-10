"""
    Make fake results to work before having experiments make
"""
import random

import numpy as np

from Utils import fixed_values, common_functions, paths


def make_fake_data() -> None:
    """
        Make fake csv
    """
    results_file = f"{paths.RESULTS_PATH}/fake_results.csv"
    with open(results_file, 'w') as f:
        f.write("DATASET;PREPROCESS;CLASSIFIER_NAME;"
                "IDX_EXTERNAL;FEATURES_NUMBER;PARAMS;"
                "METRICS_DICT\n")
    for dataset in fixed_values.DATASETS:
        _, X, y = common_functions.load_data(dataset)
        for classifier_name, classifier in fixed_values.CLASSIFIERS.items():
            if dataset not in classifier['datasets']:
                continue
            clf_to_val = classifier['clf']
            for preprocess in ['whole'] + fixed_values.PREPROCESSES:
                for idx_external in range(fixed_values.EXTERNAL_SPLITS):
                    best_params = random.choice(common_functions.get_all_permutations(classifier['param_grid']))
                    best_features_number = None if preprocess == 'whole' else random.choice(fixed_values.DIMENSION_GRID)
                    metrics_dict = {key: np.random.normal(loc=0.6, scale=0.2) for key in
                                    fixed_values.EVALUATION_METRICS}
                    csv_string = f"{dataset};{preprocess};{classifier_name};{idx_external};" \
                                 f"{best_features_number};{best_params};{metrics_dict}"
                    with open(f'{results_file}', 'a') as f:
                        f.write(f"{csv_string}\n")


if __name__ == '__main__':
    make_fake_data()
