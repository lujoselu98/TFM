"""
    useful functions to make plots
"""
from typing import Optional

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from tqdm.notebook import tqdm as nb_tqdm

from Utils import pandas_utils, fixed_values, paths

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_data_desc(tt: np.ndarray, X: pd.DataFrame, y: pd.Series,
                   title: Optional[str] = "Dataset description", y_label: Optional[str] = "dataset value",
                   save: Optional[bool] = False, save_path: Optional[str] = None,
                   ax: Optional[Axes] = None, fourier: Optional[bool] = False) -> None:
    """

    :param tt: Data to plot (time index)
    :param X: Data to plot  (features)
    :param y: Data to plot  (label)
    :param title: Title of the plot
    :param y_label: Label og y axis
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    :param fourier: set to plot fourier plots
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

    if not fourier:
        tt = tt / 4
    ax.plot(tt, mean_0, label='Class 0', color=COLORS[0])
    ax.fill_between(tt, mean_0 - std_0, mean_0 + std_0, label='Class 0 Std.', alpha=0.5, color=COLORS[0])
    ax.fill_between(tt, min_0, max_0, label='Class 0 Min./Max.', alpha=0.25, color=COLORS[0])

    ax.plot(tt, mean_1, label='Class 1', color=COLORS[1])
    ax.fill_between(tt, mean_1 - std_1, mean_1 + std_1, label='Class 1 Std.', alpha=0.5, color=COLORS[1])
    ax.fill_between(tt, min_1, max_1, label='Class 1 Min./Max.', alpha=0.25, color=COLORS[1])

    ax.legend(loc='best')
    ax.set_title(f"{title}")
    ax.set_ylabel(f"{y_label}")
    if not fourier:
        ax.set_xlabel("Lag (s)")
    else:
        ax.set_xlabel("Freq (Hz)")
        ax.set_yscale('log')

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")


def plot_fhr_uc(fhr: pd.DataFrame, uc: pd.DataFrame, y: pd.Series,
                save: Optional[bool] = False, save_path: Optional[str] = None,
                ax: Optional[Axes] = None) -> None:
    """

    :param fhr: Data to plot
    :param uc: Data to plot  (features)
    :param y: Data to plot  (label)
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    """

    if save and save_path is None:
        raise ValueError("If save is set it needs a save path")
    if not save and save_path is not None:
        raise Warning("If save is not set the save_path is ignored. Figure not saved.")

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    ax[0].plot(fhr[y == 0].mean().index, fhr[y == 0].mean().values, label='Media FHR normal')
    ax[0].plot(fhr[y == 1].mean().index, fhr[y == 1].mean().values, label='Media FHR patológico')
    ax[1].plot(uc[y == 0].mean().index, uc[y == 0].mean().values, label='Media UC normal')
    ax[1].plot(uc[y == 1].mean().index, uc[y == 1].mean().values, label='Mean UC patológico')

    ax[0].set_xlabel('Time(s)', fontsize=25)
    ax[1].set_xlabel('Time(s)', fontsize=25)
    ax[0].set_ylabel('FHR', fontsize=25)
    ax[1].set_ylabel('UC', fontsize=25)
    ax[0].legend()
    ax[1].legend()

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")


def plot_class_correlation(X, y,
                           title: Optional[str] = "Correlation with class",
                           save: Optional[bool] = False, save_path: Optional[str] = None,
                           ax: Optional[Axes] = None) -> None:
    """

    :param X: Data to plot  (features)
    :param y: Data to plot  (label)
    :param title: Title of the plot
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    """

    if save and save_path is None:
        raise ValueError("If save is set it needs a save path")
    if not save and save_path is not None:
        raise Warning("If save is not set the save_path is ignored. Figure not saved.")

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    X.corrwith(y).plot(ax=ax)

    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag')
    ax.set_title(f'{title}')

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")


def plot_class_proportion(y: pd.Series,
                          save: Optional[bool] = False, save_path: Optional[str] = None,
                          ax: Optional[Axes] = None) -> None:
    """

    :param y: Data to plot  (label)
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    """

    if save and save_path is None:
        raise ValueError("If save is set it needs a save path")
    if not save and save_path is not None:
        raise Warning("If save is not set the save_path is ignored. Figure not saved.")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    sns.barplot(x='index', y='ph', data=pd.DataFrame(y.value_counts(normalize=True)).reset_index(), ax=ax)
    ax.set_ylabel('% patterns')

    ax.set_xlabel('Class')

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")


def plot_relevance(X, y,
                   title: Optional[str] = "Distance correlation with class",
                   save: Optional[bool] = False, save_path: Optional[str] = None,
                   ax: Optional[Axes] = None) -> None:
    """

    :param X: Data to plot  (features)
    :param y: Data to plot  (label)
    :param title: Title of the plot
    :param save: Save it or not
    :param save_path: path to save
    :param ax: axis to plot or new figure
    """

    if save and save_path is None:
        raise ValueError("If save is set it needs a save path")
    if not save and save_path is not None:
        raise Warning("If save is not set the save_path is ignored. Figure not saved.")

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    X_t = X.T
    y = y.astype('float')

    candidates = X_t.index.to_list()
    relevance_vector = dict()

    for idx in nb_tqdm(candidates):
        row_vals = X_t.loc[idx]
        relevance_vector[idx] = dcor.u_distance_correlation_sqr(row_vals, y)

    ax.plot(np.array(list(relevance_vector.keys()), dtype='float64'), relevance_vector.values())
    # ax.set_xticks(np.arange(min(relevance_vector.keys()), max(relevance_vector.keys())+1, 100))
    ax.set_ylabel('Relevance with the class')
    ax.set_xlabel('Lag')
    ax.set_title(f'{title}')

    if save:
        plt.gcf().savefig(f"{save_path}.pdf")


def plot_CD_diagram(results_file: str, dataset: str, width: Optional[int] = 10) -> plt.Figure:
    """

    :param results_file: csv with results
    :param dataset: dataset to use on plot
    :param width: with of the plot. Default is 10.
    :return: The plot
    """

    from Orange import evaluation

    results_data = pd.read_csv(results_file, sep=';')
    for metric in fixed_values.EVALUATION_METRICS:
        results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data['MODEL_NAME'] = results_data['PREPROCESS'] + " + " + results_data['CLASSIFIER_NAME']
    results_data = results_data[results_data['CLASSIFIER_NAME'] != 'LRSScaler']
    results_data = results_data[['DATASET', 'MODEL_NAME', *fixed_values.EVALUATION_METRICS, 'IDX_EXTERNAL']]

    dataset_results = results_data[results_data['DATASET'] == dataset]
    models_avg_rank = {model: 0 for model in dataset_results.MODEL_NAME.unique()}

    ranks = dataset_results.IDX_EXTERNAL.unique().__len__()
    lowv, highv = 0, dataset_results.MODEL_NAME.unique().__len__(),

    for idx_external in range(ranks):
        ordered_results = dataset_results[dataset_results['IDX_EXTERNAL'] == idx_external].sort_values('AUC_SCORE',
                                                                                                       ascending=False)
        models_order = ordered_results.MODEL_NAME.to_list()
        for model in models_avg_rank.keys():
            models_avg_rank[model] += models_order.index(model)
    models_avg_rank = {key: value / ranks for key, value in models_avg_rank.items()}

    cd = evaluation.compute_CD(list(models_avg_rank.values()), 100)
    fig = evaluation.graph_ranks(list(models_avg_rank.values()), names=list(models_avg_rank.keys()),
                                 width=width, lowv=lowv, highv=highv, cd=cd)
    fig.set_suptitle(dataset)
    return fig


def plot_results(csv_file: str, metric: str, extra: Optional[str] = ''):

    """
    Make stripplot and boxplot of a csv file of results for a given metric

    :param csv_file: file with results
    :param metric: metric to plot
    :param extra: to add to save file
    """

    results_data = pd.read_csv(f"{paths.RESULTS_PATH}/{csv_file}", sep=';')
    results_data[metric] = results_data['METRICS_DICT'].apply(lambda x: pandas_utils.extract_dict(x, metric))
    results_data = results_data[['DATASET', 'CLASSIFIER_NAME', 'PREPROCESS', metric]]
    results_data['MODEL_NAME'] = results_data['PREPROCESS'] + " + " + results_data['CLASSIFIER_NAME']

    datasets = results_data['DATASET'].unique()

    stripplot_data = results_data[['MODEL_NAME', 'DATASET'] + [metric]]
    plt.figure(figsize=(30, 10))

    ax = sns.stripplot(data=stripplot_data, y=metric, x='DATASET', hue='MODEL_NAME', dodge=True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    for idx, minutes in enumerate(datasets[:-1]):
        plt.axvline(x=idx + 0.5)
    ax.set_title(f"{metric}", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Dataset', fontsize=25)
    ax.set_ylabel(metric, fontsize=25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=6)
    ax.axhline(0.5, color='red')
    plt.gcf().savefig(f"stripplot_{metric}_{extra}.pdf")

    plt.figure(figsize=(30, 10))

    ax = sns.boxplot(data=stripplot_data, y=metric, x='DATASET', hue='MODEL_NAME', dodge=True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    for idx, minutes in enumerate(datasets[:-1]):
        plt.axvline(x=idx + 0.5)
    ax.set_title(f"{metric}", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Dataset', fontsize=25)
    ax.set_ylabel(metric, fontsize=25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=6)
    ax.axhline(0.5, color='red')
    plt.gcf().savefig(f"boxplot_{metric}_{extra}.pdf")
