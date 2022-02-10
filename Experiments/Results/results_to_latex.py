"""
    Functions to parse results and to make Latex code
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

from Utils import pandas_utils, fixed_values, paths, latex


def create_latex_table(file, dataset, metric, mark_best_preprocess=False, mark_best_classifier=True,
                       style_mark='textbf'):
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
    table_lines[0] = '\\begin{adjustbox}{width=1.6\\textwidth}\n\\begin{tabular}{c|' + 'c' * len(preprocesses) + '}'
    table_lines[-1] = '\\end{tabular}\n\\end{adjustbox}'
    table_lines.append('\\caption{\\label{tab:' + metric.lower() + '_' + dataset + '} ' + metric + ' ' + dataset + '}')

    latex_table = '\n'.join(table_lines)

    if mark_best_preprocess:
        for best_preprocess_value in best_preprocess:
            latex_table = latex_table.replace(best_preprocess_value, f"\\{style_mark}{{{best_preprocess_value}}}")

    if mark_best_classifier:
        for best_classifier_value in best_classifier:
            latex_table = latex_table.replace(best_classifier_value, f"\\{style_mark}{{{best_classifier_value}}}")

    latex_table = latex_table.replace(latex.PM_STRING, latex.PM_LATEX)
    print(latex_table)


def main() -> None:
    """
        Main function
    """
    create_latex_table(f"{paths.RESULTS_PATH}/fake_results.csv", 'CC', 'BAL_ACC')


if __name__ == '__main__':
    main()
