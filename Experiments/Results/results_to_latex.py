"""
    Functions to parse results and to make Latex code
"""
from typing import Optional, List

import pandas as pd
from pylatex import Document, Command, Tabular, MultiColumn, MultiRow, Package, Table
from pylatex.base_classes import ContainerCommand
from pylatex.utils import NoEscape, bold
from tqdm import tqdm

from Utils import paths, pandas_utils, fixed_values


def main() -> None:
    """
        Main function to generate all posible exported latex documents formats
    """
    file = '22_02_22_results_main_experiment.csv'
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
    doc = Document('multirow', documentclass='report')

    doc.packages.append(Package('multirow'))
    doc.packages.append(Package('adjustbox'))
    doc.packages.append(Package('ctable'))

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


if __name__ == '__main__':
    main()
