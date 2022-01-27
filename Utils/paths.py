"""
    This file is just for defining path constants
"""

# Paths for Data
DATA_PATH = "/Data"
ORIGINAL_DATA_PATH = f"{DATA_PATH}/Original_Data"


def _print_paths() -> None:
    """
    Function just to print all paths

    :return:
    """
    print(f"{DATA_PATH=}")
    print(f"{ORIGINAL_DATA_PATH=}")


if __name__ == '__main__':
    _print_paths()
