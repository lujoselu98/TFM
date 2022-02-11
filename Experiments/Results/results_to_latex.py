"""
    Functions to parse results and to make Latex code
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from Utils import pandas_utils, fixed_values, paths, latex


def create_latex_table(file: str, dataset: str, metric: str,
                       mark_best_preprocess: Optional[bool] = False, mark_best_classifier: Optional[bool] = True,
                       style_mark: Optional[str] = 'textbf') -> str:
    """

    Create a latex table from csv file for a dataset and metric set, params for different style

    :param file: csv file with the results
    :param dataset: dataset to use
    :param metric: metric to calculate
    :param mark_best_preprocess: mark by columns
    :param mark_best_classifier: mark by rows
    :param style_mark: latex style of the mark
    :return: String of latex table
    """
    if mark_best_preprocess and mark_best_classifier:
        mark_best_preprocess = False

    metrics = fixed_values.EVALUATION_METRICS.keys()
    results_data = pd.read_csv(file, sep=';')
    results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data = results_data[['DATASET', 'CLASSIFIER_NAME', 'PREPROCESS', metric]]

    dataset_results = results_data[results_data['DATASET'] == dataset]

    preprocesses = dataset_results['PREPROCESS'].unique()
    classifiers = dataset_results['CLASSIFIER_NAME'].unique()

    p_table = pd.pivot_table(dataset_results, values=metric, index='CLASSIFIER_NAME', columns=['PREPROCESS'],
                             aggfunc=['mean', 'std'])

    table = []
    for classifier in classifiers:
        row = [classifier]
        classifier_data = p_table.loc[classifier]
        for preprocess in preprocesses:
            row.append(f"{classifier_data[('mean', preprocess)]:.3f} "
                       f"{latex.PM_STRING} "
                       f"{classifier_data[('std', preprocess)]:.3f}")
        table.append(row)

    # Mark best preprocess (best by row)
    best_preprocess = []
    for row in table:
        value_means = [float(value.split(f" {latex.PM_STRING} ")[0]) for value in row[1:]]
        idx_max = np.argmax(value_means)
        best_preprocess.append(row[idx_max + 1])

    # Mark best classifier (best by column)
    best_classifier = []
    table_T = list(zip(*table))
    for col in table_T[1:]:
        values_means = [float(value.split(f' {latex.PM_STRING} ')[0]) for value in col]
        idx_max = np.argmax(values_means)
        # noinspection PyTypeChecker
        best_classifier.append(col[idx_max])

    latex_table = str(tabulate(table, headers=preprocesses, tablefmt='latex'))

    # Style
    table_lines = latex_table.split('\n')
    table_lines = ['\t' + line for line in table_lines]
    table_lines[0] = '\\begin{tabular}{c|' + 'c' * len(preprocesses) + '}'
    table_lines[-1] = '\\end{tabular}'
    table_lines.append('\\caption{\\label{tab:' + metric.lower() + '_' + dataset + '} '
                       + fixed_values.EVALUATION_METRICS[metric]['name'] + ' ' + dataset + '}')

    latex_table = '\n'.join(table_lines)

    if mark_best_preprocess:
        for best_preprocess_value in best_preprocess:
            latex_table = latex_table.replace(best_preprocess_value, f"\\{style_mark}{{{best_preprocess_value}}}")

    if mark_best_classifier:
        for best_classifier_value in best_classifier:
            latex_table = latex_table.replace(best_classifier_value, f"\\{style_mark}{{{best_classifier_value}}}")

    latex_table = latex_table.replace(latex.PM_STRING, latex.PM_LATEX)
    return latex_table


def compose_latex(file: str, metric: str) -> str:
    """
    Create whole latex document from a csv file with results for a metric

    :param file: csv file with the results
    :param metric: metric to use
    :return: String with the latex document ready to compile
    """
    whole_latex = latex.LATEX_TABLE_BEGIN
    for dataset in fixed_values.DATASETS:
        whole_latex += '\n'.join(['\t\t' + line for line in create_latex_table(file, dataset, metric).split('\n')])
        whole_latex += latex.SPACE_BETWEEN_TABLES

    whole_latex += latex.LATEX_TABLE_END
    whole_latex = '\n'.join([latex.LATEX_HEADER, latex.LATEX_MARGINS, latex.LATEX_BEGIN, whole_latex, latex.LATEX_END])

    return whole_latex


def main() -> None:
    """
        Main function
    """
    for metric in fixed_values.EVALUATION_METRICS.keys():
        latex.save_latex(compose_latex(f"{paths.RESULTS_PATH}/results_main_experiment.csv", metric),
                         f'{metric}_results',
                         current_path=os.path.abspath(os.path.dirname(__file__)))


if __name__ == '__main__':
    main()
