"""
    Functions to make latex models rank documents
"""
import pandas as pd
# noinspection PyProtectedMember
from autorank import _util as autorank_utils
from matplotlib import pyplot as plt
from Orange import evaluation

from Utils import fixed_values, pandas_utils


def read_data_csv(csv_file: str) -> pd.DataFrame:
    """

    :param csv_file: csv file with the results
    :return: DataFrame with needed data
    """

    results_data = pd.read_csv(csv_file, sep=";")
    for metric in fixed_values.EVALUATION_METRICS:
        results_data[metric] = results_data["METRICS_DICT"].apply(
            lambda x: pandas_utils.extract_dict(x, metric)
        )
    results_data["MODEL_NAME"] = (
        results_data["PREPROCESS"] + " + " + results_data["CLASSIFIER_NAME"]
    )
    results_data = results_data[results_data["CLASSIFIER_NAME"] != "LRSScaler"]
    results_data = results_data[
        ["DATASET", "MODEL_NAME", *fixed_values.EVALUATION_METRICS, "IDX_EXTERNAL"]
    ]

    return results_data


def create_csv_diagram(result: autorank_utils.RankResult):
    """

    :param result: RankResult given by autorank library
    :return: fig with the CD diagram make by

    """
    sorted_ranks, names, groups = autorank_utils.get_sorted_rank_groups(
        result, reverse=True
    )
    cd = result.cd
    evaluation.graph_ranks(sorted_ranks.values, names=names, width=10, cd=cd)

    return plt.gcf()
