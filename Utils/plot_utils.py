"""
    useful functions to make plots
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_data_desc(tt: np.ndarray, X: pd.DataFrame, y: pd.Series,
                   title: Optional[str] = "Dataset description", y_label: Optional[str] = "dataset value",
                   save: Optional[bool] = False, save_path: Optional[str] = None,
                   ax: Optional[Axes] = None) -> None:
    """

    :param tt: Data to plot (time index)
    :param X: Data to plot  (features)
    :param y: Data to plot  (label)
    :param title: Title of the plot
    :param y_label: Label og y axis
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    """

    if save and save_path is None:
        raise ValueError("If save is set it needs a save path")
    if not save and save_path is not None:
        raise Warning("If save is not set the save_path is ignored. Figure not saved.")

    X_0 = X[y == 0].copy()
    mean_0 = X_0.mean().values
    std_0 = X_0.std().values
    max_0 = X_0.max().values
    min_0 = X_0.min().values

    X_1 = X[y == 1].copy()
    mean_1 = X_1.mean().values
    std_1 = X_1.std().values
    max_1 = X_1.max().values
    min_1 = X_1.min().values

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot(tt / 4, mean_0, label='Class 0', color=COLORS[0])
    ax.fill_between(tt / 4, mean_0 - std_0, mean_0 + std_0, label='Class 0 Std.', alpha=0.5, color=COLORS[0])
    ax.fill_between(tt / 4, min_0, max_0, label='Class 0 Min./Max.', alpha=0.25, color=COLORS[0])

    ax.plot(tt / 4, mean_1, label='Class 1', color=COLORS[1])
    ax.fill_between(tt / 4, mean_1 - std_1, mean_1 + std_1, label='Class 1 Std.', alpha=0.5, color=COLORS[1])
    ax.fill_between(tt / 4, min_1, max_1, label='Class 1 Min./Max.', alpha=0.25, color=COLORS[1])

    ax.legend(loc='best')
    ax.set_title(f"{title}")
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel("Lag (s)")

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")
