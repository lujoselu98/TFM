"""
    This file is just for defining path constants
"""
import os

# Root of project get the parent folder
PROJECT_PATH: str = os.path.abspath(os.path.dirname(__file__))
PROJECT_PATH = '/'.join(PROJECT_PATH.split('\\')[:-1])

# Paths for Data
DATA_PATH: str = f"{PROJECT_PATH}/Data"
ORIGINAL_DATA_PATH: str = f"{DATA_PATH}/Original_Data"
CLEAN_DATA_PATH: str = f"{DATA_PATH}/Clean_Data"
CC_DATA_PATH: str = f"{DATA_PATH}/cc_data"
CDCOR_DATA_PATH: str = f"{DATA_PATH}/cdcor_data"
FFT_DATA_PATH: str = f"{DATA_PATH}/fft_Data"

# Path for Plots
PLOTS_PATH: str = f"{PROJECT_PATH}/Plots"
ORIGINAL_DATA_PLOTS: str = f"{PLOTS_PATH}/Original_Data"
CC_DATA_PLOTS: str = f"{PLOTS_PATH}/cc_data"
CDCOR_DATA_PLOTS: str = f"{PLOTS_PATH}/cdcor_data"
FFT_DATA_PLOTS: str = f"{PLOTS_PATH}/fft_Data"

# Paths for preprocessing paths
MRMR_PATH = f"{PROJECT_PATH}/Preprocessing/mRMR/mRMR_files"
FPCA_PATH = f"{PROJECT_PATH}/Preprocessing/FPCA/FPCA_files"
PLS_PATH = f"{PROJECT_PATH}/Preprocessing/PLS/PLS_files"

# Paths to save results
RESULTS_PATH = f"{PROJECT_PATH}/Experiments/Results"
CLASSIFIERS_PATH = f"{PROJECT_PATH}/Experiments/Classifiers"
LATEX_PATH = f"{PROJECT_PATH}/Experiments/Latex"


def _print_paths() -> None:
    """
    Function just to print all paths
    """
    print(f"PROJECT_PATH={PROJECT_PATH}")
    print()

    print(f"DATA_PATH={DATA_PATH}")
    print(f"ORIGINAL_DATA_PATH={ORIGINAL_DATA_PATH}")
    print(f"CC_DATA_PATH={CC_DATA_PATH}")
    print(f"CDCOR_DATA_PATH={CDCOR_DATA_PATH}")
    print(f"FFT_DATA_PATH={FFT_DATA_PATH}")
    print()

    print(f"PLOTS_PATH={PLOTS_PATH}")
    print(f"ORIGINAL_DATA_PLOTS={ORIGINAL_DATA_PLOTS}")
    print(f"CC_DATA_PLOTS={CC_DATA_PLOTS}")
    print(f"CDCOR_DATA_PLOTS={CDCOR_DATA_PLOTS}")
    print(f"FFT_DATA_PLOTS={FFT_DATA_PLOTS}")
    print()

    print(f"MRMR_PATH={MRMR_PATH}")
    print(f"FPCA_PATH={FPCA_PATH}")
    print(f"PLS_PATH={PLS_PATH}")
    print(f"CLASSIFIERS_PATH={CLASSIFIERS_PATH}")
    print(f"LATEX_PATH={LATEX_PATH}")
    print()


if __name__ == '__main__':
    _print_paths()
