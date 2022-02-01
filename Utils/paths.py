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
CLEAN_CUT_DATA_PATH: str = f"{DATA_PATH}/Clean_Data"
CC_DATA_PATH: str = f"{DATA_PATH}/cc_data"
CDCOR_DATA_PATH: str = f"{DATA_PATH}/cdcor_data"

# Path for Plots
PLOTS_PATH: str = f"{PROJECT_PATH}/Plots"
ORIGINAL_DATA_PLOTS: str = f"{PLOTS_PATH}/Original_Data"
CC_DATA_PLOTS: str = f"{PLOTS_PATH}/cc_data"
CDCOR_DATA_PLOTS: str = f"{PLOTS_PATH}/cdcor_data"


def _print_paths() -> None:
    """
    Function just to print all paths
    """
    print(f"{PROJECT_PATH=}")
    print()

    print(f"{DATA_PATH=}")
    print(f"{ORIGINAL_DATA_PATH=}")
    print(f"{CLEAN_CUT_DATA_PATH=}")
    print(f"{CC_DATA_PATH=}")
    print(f"{CDCOR_DATA_PATH=}")
    print()

    print(f"{PLOTS_PATH=}")
    print(f"{ORIGINAL_DATA_PLOTS=}")
    print(f"{CC_DATA_PLOTS=}")
    print(f"{CDCOR_DATA_PLOTS=}")


if __name__ == '__main__':
    _print_paths()
