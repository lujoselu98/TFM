"""
    Functions to parse results and to make Latex code
"""
import itertools
from typing import Optional, List, Dict

import matplotlib
import numpy as np
import pandas as pd
from pylatex import Document, Command, Tabular, MultiColumn, MultiRow, Package, Table
from pylatex.base_classes import ContainerCommand
from pylatex.utils import NoEscape, bold
from scipy import stats
from tqdm import tqdm

from Utils import paths, pandas_utils, fixed_values


def old_main() -> None:
    """
        Main function to generate all posible exported latex documents formats
    """
    file = '28_02_22_results_main_experiment.csv'
    progress_bar = tqdm(['classifiers', 'datasets'], total=4)
    for out in progress_bar:
        progress_bar.set_description(f"Generating latex by {out}")
        for mark_rows in [True, False]:

            progress_bar.set_postfix({'mark': f"{'row' if mark_rows else 'col'}"})

            out_file = f"{paths.LATEX_PATH}/results_by_{out}_{'row' if mark_rows else 'col'}"
            create_latex_document(csv_file=file, out_file=out_file, out=out, mark_rows=mark_rows)

            if progress_bar.last_print_n < progress_bar.total:
                progress_bar.update(1)
            else:
                progress_bar.close()


def main() -> None:
    """
        Main function to export latex colors
    """
    # file = '28_02_22_results_main_experiment.csv'
    # file = '11_03_22_results_main_exxperiment_no_outliers.csv'
    file = 'results_1647440481.7991931_main_experiment.csv'
    out_file = f"{paths.LATEX_PATH}/color_map_test"

    create_latex_color_document(file, out_file, color_map='YlGn')


def create_latex_document(csv_file: str, out_file: str, out: str = 'classifiers', mark_rows: Optional[bool] = True,
                          clean_tex: Optional[bool] = False) -> None:
    """

    Export results from csv to Latex in different formats

    :param csv_file: csv file with the experiments results
    :param out_file: pdf file to generate
    :param out: more outside values in table ['classifiers', 'datasets']
    :param mark_rows: true to mark by rows, false to mark columns by block
    :param clean_tex: true to clean tex after pdf generation, default is False
    """
    assert csv_file.endswith('csv')
    assert out in ['classifiers', 'datasets']

    doc = _latex_preamble()

    for metric in fixed_values.EVALUATION_METRICS:
        data = _get_data_from_csv(f"{paths.RESULTS_PATH}/{csv_file}", metric)
        preprocesses = data.columns.get_level_values(1).unique()
        classifiers = data.index.get_level_values(0).unique()
        datasets = data.index.get_level_values(1).unique()

        size_box = ContainerCommand(arguments=[NoEscape(r'\textwidth'), '!'])
        size_box.latex_name = 'resizebox'

        table = Table()

        tabular = Tabular('cc|c|c|c|c')
        blank_columns = ['' for _ in range(tabular.width - len(preprocesses))]
        tabular.add_row((*blank_columns, *preprocesses))
        tabular.append(Command('specialrule', arguments=['.2em', '.1em', '.1em']))

        if out == 'classifiers':
            first_level = classifiers
            second_level = datasets
        else:
            first_level = datasets
            second_level = classifiers

        best_datas = dict()
        for classifier in classifiers:
            classifier_results = data.loc[classifier]['mean']
            best_datas[classifier] = classifier_results.idxmax()

        best_classifiers = dict()
        for dataset in datasets:
            dataset_results = data.swaplevel(1, 0).loc[dataset]['mean']
            best_classifiers[dataset] = dataset_results.idxmax()

        for it_first_level in first_level:

            if out == 'classifiers':
                list_of_best = best_datas[it_first_level]
            else:
                list_of_best = best_classifiers[it_first_level]

            for i, it_second_level in enumerate(second_level):
                # it_first_level == datasets, it_second_level == classifier, list_of_best == best_classifiers
                if (it_second_level, it_first_level) in data.index:
                    row_data = data.loc[(it_second_level, it_first_level)]
                    if mark_rows:
                        row_data = _create_latex_row(row_data)
                    else:
                        row_data = _create_latex_row(row_data,
                                                     list_of_best[list_of_best == it_second_level].index.to_list())

                # it_first_level == datasets, it_second_level == classifier, list_of_best == best_datas
                elif (it_first_level, it_second_level) in data.index:
                    row_data = data.loc[(it_first_level, it_second_level)]
                    if mark_rows:
                        row_data = _create_latex_row(row_data)
                    else:
                        row_data = _create_latex_row(row_data,
                                                     list_of_best[list_of_best == it_second_level].index.to_list())

                else:
                    if out == 'datasets':
                        continue
                    row_data = ['No converge' for _ in range(len(preprocesses))]
                if i == 0:
                    tabular.add_row(
                        (
                            MultiColumn(1, align='c|', data=MultiRow(3, data=it_first_level)),
                            it_second_level, *row_data
                        )
                    )
                else:
                    tabular.add_row(
                        (
                            MultiColumn(1, align='c|', data=''),
                            it_second_level, *row_data
                        )
                    )
                if i != len(second_level) - 1:
                    tabular.add_hline(start=2, end=len(preprocesses) + 2)
                else:
                    tabular.append(Command('specialrule', arguments=['.2em', '.1em', '.1em']))

        caption = Command('caption', f"Tabla comparativa en {fixed_values.EVALUATION_METRICS[metric]['name']}")

        size_box.append(tabular)
        table.append(size_box)
        table.append(caption)
        doc.append(table)
    doc.generate_pdf(out_file, compiler='pdflatex', clean_tex=clean_tex)
    # doc.generate_tex(f"{paths.LATEX_PATH}/test_pyLatex")
    # print(doc.dumps())


def create_latex_color_document(csv_file: str, out_file: str, clean_tex: Optional[bool] = False,
                                color_map: Optional[str] = 'YlGn') -> None:
    """

    Export results from csv to Latex in different formats

    :param color_map: color map of the table
    :param csv_file: csv file with the experiments results
    :param out_file: pdf file to generate
    :param clean_tex: true to clean tex after pdf generation, default is False
    """
    assert csv_file.endswith('csv')

    doc = _latex_preamble()

    cmap = matplotlib.cm.get_cmap(color_map)
    # rgb_colors_24 = cmap([i / 24 for i in range(24)])  # 24 for FFT 4 pre x 6 clf
    rgb_colors_24 = cmap([i * (0.9 - 0.1) / 24 for i in range(24)])  # 24 for FFT 4 pre x 6 clf
    # rgb_colors_20 = cmap([i / 20 for i in range(20)])  # 12 for CC and DCOR 4 pre x 6 clf
    rgb_colors_20 = cmap([i * (0.9 - 0.1) / 20 for i in range(20)])  # 12 for CC and DCOR 4 pre x 6 clf

    for i, color in enumerate(rgb_colors_24[::-1]):
        doc.preamble.append(NoEscape(r"\definecolor{green_" + str(i) + "}{rgb}{"
                                     + str(color[0]) + "," + str(color[1]) + "," + str(color[2]) +
                                     "}"))

    for i, color in enumerate(rgb_colors_20[::-1]):
        doc.preamble.append(NoEscape(r"\definecolor{green_20_" + str(i) + "}{rgb}{"
                                     + str(color[0]) + "," + str(color[1]) + "," + str(color[2]) +
                                     "}"))

    for metric in fixed_values.EVALUATION_METRICS:
        data = _get_data_from_csv(f"{paths.RESULTS_PATH}/{csv_file}", metric)
        preprocesses = data.columns.get_level_values(1).unique()
        datasets = ['CC', 'DCOR']  # data.index.get_level_values(1).unique()

        classifiers = dict()
        for dataset in datasets:
            classifiers[dataset] = data.loc[pd.IndexSlice[:, dataset], :].index.get_level_values(0)

        size_box = ContainerCommand(arguments=[NoEscape(r'\textwidth'), '!'])
        size_box.latex_name = 'resizebox'

        table = Table()

        tabular = Tabular('cc|c|c|c|c')
        blank_columns = ['' for _ in range(tabular.width - len(preprocesses))]
        tabular.add_row((*blank_columns, *preprocesses))
        tabular.append(Command('specialrule', arguments=['.2em', '.1em', '.1em']))

        colors_order = dict()
        for dataset in datasets:
            if dataset == 'FFT':
                colors = np.array([f'green_{i}' for i in range(24)]).reshape(
                    (len(classifiers[dataset]), len(preprocesses)))
            else:
                colors = np.array([f'green_20_{i}' for i in range(20)]).reshape(
                    (len(classifiers[dataset]), len(preprocesses)))
            colors_order[dataset] = _get_color_matrix(data.loc[pd.IndexSlice[:, dataset], :]['mean'], colors)
        for block, dataset in enumerate(datasets):
            for row, classifier in enumerate(classifiers[dataset]):
                # Hay classifiers que no se usan para todos los classifiers
                if (classifier, dataset) in data.index:
                    row_data = data.loc[(classifier, dataset)]

                    start_brace = "{"
                    end_brace = "}"

                    row_colors = colors_order[dataset][row]
                    row_data = [
                        NoEscape(f" \cellcolor{start_brace}{row_colors[i]}{end_brace}"
                                 f"{row_data[('mean', preprocess)]:.3f} $\pm$ {row_data[('std', preprocess)]:.3f}")
                        for i, preprocess in enumerate(preprocesses)]
                else:
                    continue

                if row == 0:
                    tabular.add_row(
                        (
                            MultiColumn(1, align='c|', data=MultiRow(3, data=dataset)),
                            classifier, *row_data
                        )
                    )
                else:
                    tabular.add_row(
                        (
                            MultiColumn(1, align='c|', data=''),
                            classifier, *row_data
                        )
                    )

                if row != len(classifiers[dataset]) - 1:
                    tabular.add_hline(start=2, end=len(preprocesses) + 2)
                else:
                    tabular.append(Command('specialrule', arguments=['.2em', '.1em', '.1em']))

        caption = Command('caption', f"Tabla comparativa en {fixed_values.EVALUATION_METRICS[metric]['name']}")
        size_box.append(tabular)
        table.append(size_box)
        table.append(caption)
        doc.append(table)
    doc.generate_pdf(out_file, compiler='pdflatex', clean_tex=clean_tex)
    # doc.generate_tex(out_file)
    # print(doc.dumps())


def _get_color_matrix(data: pd.DataFrame, colors: np.ndarray) -> np.ndarray:
    row_order, col_order = [], []

    index_list = data.index.values.tolist()
    columns_list = data.columns.values.tolist()
    sort_values = np.sort(data.values.flatten())[::-1]

    for value in sort_values:
        for col in data.columns:
            if value in data[col].values:
                row_order.append(index_list.index(data.index[data[col] == value]))
                col_order.append(columns_list.index(col))

    colors_order = np.empty(dtype='<U12', shape=(len(index_list), len(columns_list)))
    order = [(r, c) for r, c in zip(row_order, col_order)]
    counter = 0
    for row in range(colors_order.shape[0]):
        for col in range(colors_order.shape[1]):
            colors_order[order[counter]] = colors[row, col]
            counter += 1

    return colors_order


def _latex_preamble() -> Document:
    """
        Create doc of latex and preamble
    """
    doc = Document('multirow', documentclass='report')

    doc.packages.append(Package('multirow'))
    doc.packages.append(Package('adjustbox'))
    doc.packages.append(Package('ctable'))
    doc.packages.append(Package('xcolor'))
    doc.packages.append(Package('colortbl'))

    doc.preamble.append(Command('title', 'Resultados Experimentos DAHFI'))
    doc.preamble.append(Command('author', 'Jose Luis Lavado'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    doc.preamble.append(NoEscape(r"\textwidth = 16truecm"))
    doc.preamble.append(NoEscape(r"\textheight = 25truecm"))
    doc.preamble.append(NoEscape(r"\oddsidemargin = -20pt"))
    doc.preamble.append(NoEscape(r"\evensidemargin = 5pt"))
    doc.preamble.append(NoEscape(r"\topmargin = -2truecm"))

    doc.append(NoEscape(r'\maketitle'))
    doc.append(NoEscape(r'\renewcommand{\arraystretch}{1.2}'))

    return doc


def _create_latex_row(data: pd.DataFrame, mark_preprocess: Optional[List] = None) -> List:
    """

    Auxiliar function to create each row of latex table with the bold mark
    :param data: data of the row
    :param mark_preprocess: list tof wich data to mark,leave None to be the max of data
    :return: list of formate data to pylatex to insert into a row
    """
    preprocesses = data.index.get_level_values(1).unique()

    if mark_preprocess is None:
        mark_preprocess = [data['mean'].idxmax()]

    row_latex_data = []

    for preprocess in preprocesses:
        if preprocess in mark_preprocess:
            row_latex_data.append(bold(NoEscape(
                f"{data[('mean', preprocess)]:.3f} $\pm$ {data[('std', preprocess)]:.3f}")))
        else:
            row_latex_data.append(
                NoEscape(f"{data[('mean', preprocess)]:.3f} $\pm$ {data[('std', preprocess)]:.3f}"))
    return row_latex_data


def _get_significant_order(csv_file: str, dataset: str, metric: str, alpha: Optional[float] = 0.01) -> Dict[int, List]:
    """

    Method for calculate the stadistically significant rank of models using Wilcoxon, one dataset and one metric

    :param csv_file: results file
    :param dataset: dataset to filter
    :param metric: metric to filter
    :param alpha: significant level
    :return: rank dict (rank, models)
    """
    results_data = pd.read_csv(csv_file, sep=';')
    results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data['MODEL_NAME'] = results_data['PREPROCESS'] + " + " + results_data['CLASSIFIER_NAME']
    results_data = results_data[results_data['CLASSIFIER_NAME'] != 'LRSScaler']
    results_data = results_data[['DATASET', 'MODEL_NAME', metric]]
    dataset_results = results_data[results_data['DATASET'] == dataset]

    # Rank all models
    dataset_metrics = np.zeros(len(dataset_results.MODEL_NAME.unique()))
    for i, model in enumerate(dataset_results.MODEL_NAME.unique()):
        dataset_model_results = dataset_results[dataset_results['MODEL_NAME'] == model][metric]
        dataset_metrics[i] = dataset_model_results.mean()
    if any(np.diff(dataset_metrics) == 0):
        raise ValueError(f"Ties on dataset {dataset} with metric {metric}")
    model_ranks = dataset_results.MODEL_NAME.unique()[np.argsort(dataset_metrics)[::-1]]

    models = dataset_results.MODEL_NAME.unique()

    # Calculate p_value for all combinations of models (np.nan default)
    p_values = pd.DataFrame(0, index=models, columns=models)
    for model in models:
        p_values.loc[model, model] = np.nan

    for model_1, model_2 in itertools.permutations(models, 2):
        results_model_1 = dataset_results[dataset_results.MODEL_NAME == model_1][metric]
        results_model_2 = dataset_results[dataset_results.MODEL_NAME == model_2][metric]
        p_values.loc[model_1, model_2] = stats.wilcoxon(results_model_1.values, results_model_2.values)[1]

    p_values = p_values.loc[model_ranks, model_ranks]

    p_values.to_excel(f'p_values_{dataset}_{metric}.xls')

    # Finally rank the models
    returned_rank = dict()
    rank = 0
    ranked = []
    for model in model_ranks:
        if model in ranked:
            continue
        rank_models = p_values.index[p_values.loc[model] > alpha].to_list()
        rank_models = [model for model in rank_models if model not in ranked]
        rank_models.append(model)
        returned_rank[rank] = rank_models[::-1]
        ranked.extend(rank_models)
        rank += 1

    return returned_rank


def _get_data_from_csv(csv_file: str, metric: str) -> pd.DataFrame:
    """

    Auxiliary function to get data from csv file into pivot table

    :param csv_file: csv file with the results of experiments
    :param metric: metric to give on the pivot table
    :return: Pivot table with the data used to generate latex
    """
    assert csv_file.endswith(".csv")
    assert metric in fixed_values.EVALUATION_METRICS

    results_data = pd.read_csv(csv_file, sep=';')
    results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data = results_data[['DATASET', 'CLASSIFIER_NAME', 'PREPROCESS', metric]]

    results_pivot_table = pd.pivot_table(
        results_data,
        values=metric, index=['CLASSIFIER_NAME', 'DATASET'], columns=['PREPROCESS'],
        aggfunc=['mean', 'std']
    )

    return results_pivot_table


def test_significance_ranking_main() -> None:
    for dataset in fixed_values.DATASETS:
        for metric in fixed_values.EVALUATION_METRICS:
            ranking = _get_significant_order('28_02_22_results_main_experiment.csv', dataset, metric, alpha=0.01)
            print(f"{dataset} {metric}")
            for rank, models in ranking.items():
                print(rank, models)
            print("*" * 10)


if __name__ == '__main__':
    main()
    # test_significance_ranking_main()
