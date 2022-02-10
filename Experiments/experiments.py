"""
    Main File to do Different Experiments and save the results
"""
from tqdm import tqdm

from Utils import fixed_values


def main_experiment() -> None:
    """Function to made the main experiment"""

    progress_bar = tqdm(fixed_values.DATASETS,
                        total=len(fixed_values.DATASETS) * len(fixed_values.CLASSIFIERS) * len(
                            fixed_values.PREPROCESSES) * fixed_values.EXTERNAL_SPLITS)

