"""
    Functions and constants used to export results to Latex
"""
import os
import subprocess

from Utils import paths

PM_STRING = "pms"
PM_LATEX = "$\\pm$"

LATEX_HEADER = (
    "\\documentclass[]{report}\n"
    "\\usepackage{multirow}\n"
    "\\usepackage{booktabs}\n"
    "\\usepackage{lscape}\n"
    "\\usepackage{adjustbox}\n"
    "\\title{Resultados Experimentos DAHFI}\n"
    "\\author{Jose Luis Lavado}\n"
)

LATEX_MARGINS = (
    "\\textwidth = 16truecm\n"
    "\\textheight = 25truecm\n"
    "\\oddsidemargin =-20pt\n"
    "\\evensidemargin = 5pt\n"
    "\\topmargin=-2truecm\n"
)

LATEX_BEGIN = "\\begin{document}\n" "\\maketitle\n"

LATEX_TABLE_BEGIN = (
    "\t\\begin{table}\n"
    "\t\t\\centering\n"
    "\t\t\\setlength{\\aboverulesep}{0pt}\n"
    "\t\t\\setlength{\\belowrulesep}{0pt}\n"
)

SPACE_BETWEEN_TABLES = "\n\t\t\\vspace*{2cm}\n"


LATEX_TABLE_END = "\t\\end{table}\n"

LATEX_END = "\\end{document}"


def save_latex(latex: str, file_name: str, current_path: str) -> None:
    """
    Write latex string into latex file, compile it and clean auxiliar files
    :param latex: latex string
    :param file_name: filename to save
    :param current_path: path from where it is executed
    """

    # Write into .tex file
    with open(f"{paths.LATEX_PATH}/{file_name}.tex", "w") as f:
        f.writelines(latex)

    # Compile to pdf with pdfLatex
    subprocess.run(
        ["pdfLatex", f"{paths.LATEX_PATH}/{file_name}.tex"], stdout=subprocess.DEVNULL
    )

    # Remove auxiliary files
    os.remove(f"{current_path}/{file_name}.log")
    os.remove(f"{current_path}/{file_name}.aux")

    # Move .pdf to correct path
    os.replace(f"{current_path}/{file_name}.pdf", f"{paths.LATEX_PATH}/{file_name}.pdf")
