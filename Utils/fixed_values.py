"""
    Common shared values
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASETS = ['CC', 'DCOR', 'FFT']
# DATASETS = ['FFT']

OUTLIERS_IDX = np.array(
    [1282, 1027, 1286, 1035, 1291, 1039, 1296, 1297, 1043, 1044, 1045,
     1302, 1047, 1301, 1049, 1306, 1052, 1054, 1055, 1312, 1311, 1310,
     1313, 1059, 1278, 1318, 1065, 1066, 1072, 1073, 1330, 1080, 1338,
     1086, 1087, 1345, 1350, 1357, 1358, 1108, 1366, 1372, 1374, 1120,
     1121, 1380, 1131, 1392, 1137, 1397, 1145, 1147, 1407, 1156, 1157,
     1414, 1415, 1164, 1422, 1424, 1428, 1174, 1431, 1432, 1434, 1181,
     1440, 1189, 1448, 1196, 1197, 1198, 1199, 1458, 1210, 1466, 1469,
     1470, 1228, 1235, 1495, 1239, 2012, 1501, 2013, 2021, 2024, 1261,
     1009, 2035, 1268, 1013, 1270, 1018, 1019, 1022]
)

CC_OUTLIERS_IDX = np.array([1009, 1027, 1035, 1055, 1197, 1198, 1302, 1312, 2024])

DCOR_OUTLIERS_IDX = np.array(
    [1019, 1022, 1045, 1047, 1066, 1086, 1120, 1147, 1157, 1181, 1189,
     1198, 1306, 1311, 1318, 1330, 1350, 1397, 1415, 1440, 1458, 1469,
     1495, 2012, 2024, 2035]
)

FFT_OUTLIERS_IDX = np.array(
    [1043, 1065, 1073, 1137, 1164, 1174, 1196, 1199, 1228, 1268, 1297,
     1313, 1338, 1407, 1422, 1428, 1431, 1501, 2013, 2021]
) # cutoff_factor=5

DATASET_OUTLIERS = {
    'CC': CC_OUTLIERS_IDX,
    'DCOR': DCOR_OUTLIERS_IDX,
    'FFT': FFT_OUTLIERS_IDX
}


EXTERNAL_SPLITS = 10
EXTERNAL_SPLITS_SHUFFLE = 100
EXTERNAL_TEST_SIZE = 0.25

INTERNAL_SPLITS = 10

# PREPROCESSES = ['mRMR', 'PCA', 'PLS']
# PREPROCESSES = ['PCA', 'PLS']
PREPROCESSES = ['mRMR']

DIMENSION_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100]
MAX_DIMENSION = max(DIMENSION_GRID)

# Slower to faster
CLASSIFIERS = {
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
}

DUMMY_CLASSIFIER = {
    'clf': DummyClassifier(strategy='constant', constant=1),
    'param_grid': {},
    'evaluate_score': 'predict_proba',
    'datasets': DATASETS
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
