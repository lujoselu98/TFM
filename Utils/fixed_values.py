"""
    Common shared values
"""

EXTERNAL_SPLITS = 10
INTERNAL_SPLITS = 10

PREPROCESSES = ['mRMR', 'PCA', 'PLS']

DIMENSION_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100]
MAX_DIMENSION = max(DIMENSION_GRID)


def _print_values() -> None:
    print(f"EXTERNAL_SPLITS: {EXTERNAL_SPLITS}")
    print(f"INTERNAL_SPLITS: {INTERNAL_SPLITS}")

    print(f"PREPROCESSES: {PREPROCESSES}")
    print(f"DIMENSION_GRID: {DIMENSION_GRID}")
    print(f"MAX_DIMENSION: {MAX_DIMENSION}")


if __name__ == '__main__':
    _print_values()
