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


def _print_paths() -> None:
    """
    Function just to print all paths
    """
    print(f"{PROJECT_PATH=}")
    print()

    print(f"{DATA_PATH=}")
    print(f"{ORIGINAL_DATA_PATH=}")


if __name__ == '__main__':
    _print_paths()
    print()
