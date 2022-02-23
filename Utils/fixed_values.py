"""
    Common shared values
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASETS = ['CC', 'DCOR', 'FFT']

EXTERNAL_SPLITS = 20  # 10
INTERNAL_SPLITS = 10

PREPROCESSES = ['mRMR', 'PCA', 'PLS']

DIMENSION_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100]
MAX_DIMENSION = max(DIMENSION_GRID)

CLASSIFIERS = {
    'LR': {
        'clf': LogisticRegression(penalty='l1', solver='liblinear', random_state=0, class_weight='balanced',
                                  max_iter=1000),
        'param_grid': {
            'C': np.logspace(-3, 2, 6)
        },
        'evaluate_score': 'predict_proba',
        'datasets': DATASETS
    },
    'LRSScaler': {
        'clf': make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', random_state=0,
                                                                  class_weight='balanced',
                                                                  max_iter=1000)),
        'param_grid': {
            'logisticregression__C': np.logspace(-3, 2, 6)
        },
        'evaluate_score': 'predict_proba',
        'datasets': ['FFT']
    },
    # 'LRmMScaler': {
    #     'clf': make_pipeline(MinMaxScaler(), LogisticRegression(penalty='l1', solver='liblinear', random_state=0,
    #                                                             class_weight='balanced',
    #                                                             max_iter=1000)),
    #     'param_grid': {
    #         'logisticregression__C': np.logspace(-3, 2, 6)
    #     },
    #     'evaluate_score': 'predict_proba',
    #     'datasets': ['FFT']
    # },
    'KNN': {
        'clf': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': np.arange(3, 30, 2),
        },
        'evaluate_score': 'predict_proba',
        'datasets': DATASETS
    },
    'KNNSScaler': {
        'clf': make_pipeline(StandardScaler(), KNeighborsClassifier()),
        'param_grid': {
            'kneighborsclassifier__n_neighbors': np.arange(3, 30, 2),
        },
        'evaluate_score': 'predict_proba',
        'datasets': DATASETS
    },
    # 'KNNmMScaler': {
    #     'clf': make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
    #     'param_grid': {
    #         'kneighborsclassifier__n_neighbors': np.arange(3, 30, 2),
    #     },
    #     'evaluate_score': 'predict_proba',
    #     'datasets': DATASETS
    # },
    'SVC': {
        'clf': SVC(kernel='rbf', random_state=0, class_weight='balanced'),
        'param_grid': {
            'gamma': np.logspace(-3, 3, 7),
            'C': np.logspace(-3, 3, 7),
        },
        'evaluate_score': 'decision_function',
        'datasets': DATASETS
    },
    'SVCSScaler': {
        'clf': make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0, class_weight='balanced')),
        'param_grid': {
            'svc__gamma': np.logspace(-3, 3, 7),
            'svc__C': np.logspace(-3, 3, 7),
        },
        'evaluate_score': 'decision_function',
        'datasets': DATASETS
    },
    # 'SVCmMScaler': {
    #     'clf': make_pipeline(MinMaxScaler(), SVC(kernel='rbf', random_state=0, class_weight='balanced')),
    #     'param_grid': {
    #         'svc__gamma': np.logspace(-3, 3, 7),
    #         'svc__C': np.logspace(-3, 3, 7),
    #     },
    #     'evaluate_score': 'decision_function',
    #     'datasets': DATASETS
    # },
}

VALIDATION_METRIC = balanced_accuracy_score

EVALUATION_METRICS = {
    'BAL_ACC': {
        'function': balanced_accuracy_score,
        'values': 'predictions',
        'name': 'Balanced Accuracy',
    },
    'AUC_SCORE': {
        'function': roc_auc_score,
        'values': 'scores',
        'name': 'Area bajo la curva roc',
    },
}


def _print_values() -> None:
    print(f"DATASETS: {DATASETS}")

    print(f"EXTERNAL_SPLITS: {EXTERNAL_SPLITS}")
    print(f"INTERNAL_SPLITS: {INTERNAL_SPLITS}")

    print(f"PREPROCESSES: {PREPROCESSES}")
    print(f"DIMENSION_GRID: {DIMENSION_GRID}")
    print(f"MAX_DIMENSION: {MAX_DIMENSION}")

    print(f"CLASSIFIERS: {CLASSIFIERS}")
    print(f"VALIDATION_METRIC: {VALIDATION_METRIC}")
    print(f"EVALUATION_METRICS: {EVALUATION_METRICS}")


if __name__ == '__main__':
    _print_values()
