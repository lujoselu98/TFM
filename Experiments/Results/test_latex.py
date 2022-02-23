"""
    Just to test python to Latex functions
"""

import pandas as pd
from pylatex import Document, Command, Tabular, MultiColumn, MultiRow, Package, Table
from pylatex.base_classes import ContainerCommand
from pylatex.utils import NoEscape, bold

from Utils import paths, pandas_utils, fixed_values


def main_test():
    file = '22_02_22_results_main_experiment.csv'

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
        data = get_data_from_csv(f"{paths.RESULTS_PATH}/{file}", metric)
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

        out = 'classifiers'
        if out == 'classifiers':
            first_level = classifiers
            second_level = datasets
        else:
            first_level = datasets
            second_level = classifiers

        for it_first_level in first_level:
            counter = 0
            for i, it_second_level in enumerate(second_level):
                if (it_second_level, it_first_level) in data.index:
                    row_data = data.loc[(it_second_level, it_first_level)]
                    row_data = create_latex_row(row_data)

                elif (it_first_level, it_second_level) in data.index:
                    row_data = data.loc[(it_first_level, it_second_level)]
                    row_data = create_latex_row(row_data)
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
    doc.generate_pdf(f"{paths.LATEX_PATH}/test_pyLatex", compiler='pdflatex', clean_tex=False)
    # doc.generate_tex(f"{paths.LATEX_PATH}/test_pyLatex")
    # print(doc.dumps())


def create_latex_row(data):
    preprocesses = data.index.get_level_values(1).unique()

    best_preprocess = data['mean'].idxmax()
    row_latex_data = []

    for preprocess in preprocesses:
        if preprocess == best_preprocess:
            row_latex_data.append(bold(NoEscape(
                f"{data[('mean', preprocess)]:.3f} $\pm$ {data[('std', preprocess)]:.3f}")))
        else:
            row_latex_data.append(
                NoEscape(f"{data[('mean', preprocess)]:.3f} $\pm$ {data[('std', preprocess)]:.3f}"))
    return row_latex_data


def get_data_from_csv(file, metric):
    results_data = pd.read_csv(file, sep=';')
    results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data = results_data[['DATASET', 'CLASSIFIER_NAME', 'PREPROCESS', metric]]

    results_pivot_table = pd.pivot_table(
        results_data,
        values=metric, index=['CLASSIFIER_NAME', 'DATASET'], columns=['PREPROCESS'],
        aggfunc=['mean', 'std']
    )

    return results_pivot_table


if __name__ == '__main__':
    main_test()
