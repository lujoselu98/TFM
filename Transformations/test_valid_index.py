"""
    Just to test first/last valid index implementations

"""
import math
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Utils import paths


def valid_index(signal_data, equal_threshold=4, threshold=4 * 2, kind='first'):
    if kind == 'last':
        signal_data = signal_data[::-1]
    elif kind == 'first':
        signal_data = signal_data[signal_data.first_valid_index():]
    anterior = signal_data.values[0]
    nans = 0
    constants = 0
    differentes = 0
    return_idx = np.nan
    for idx, value in signal_data.iteritems():
        print(f"{idx} {value}", end='\t')
        if math.isnan(value):
            print("NAN", end='\t')
            nans += 1
            if nans >= threshold:
                print("RESET", end='\t')
                differentes = 0
                return_idx = np.nan
            constants = 0
        elif anterior == value:
            print("EQUAL", end='\t')
            constants += 1
            if constants >= equal_threshold:
                print("RESET", end='\t')
                differentes = 0
                return_idx = np.nan
            nans = 0
        elif value != anterior:
            print("DIFFERENT", end='\t')
            if differentes == 0:
                print("NEW RETURN INDEX", end='\t')
                return_idx = idx
            differentes += 1
            if differentes >= threshold:
                print()
                return return_idx
            nans = 0
            constants = 0
        anterior = value
        print(f"{nans} {constants} {differentes} {return_idx}", end='\t')
        print()

    return return_idx


def get_data():
    fhr = pd.read_pickle(f'{paths.ORIGINAL_DATA_PATH}/fhr_ctu-chb.pickle')
    fhr.columns = fhr.columns.astype('float')

    uc = pd.read_pickle(f'{paths.ORIGINAL_DATA_PATH}/uc_ctu-chb.pickle')
    uc.columns = uc.columns.astype('float')

    assert fhr.columns.dtype == float
    assert uc.columns.dtype == float

    clinical = pd.read_pickle(f'{paths.ORIGINAL_DATA_PATH}/clinical_ctu-chb.pickle')

    y = clinical['ph'].apply(lambda x: 0 if x >= 7.2 else 1)

    print(fhr.shape, uc.shape, y.shape
          )

    abnormal_curves = np.array([1104, 1119, 1130, 1134, 1149, 1155, 1158, 1186, 1188, 1258, 1327,
                                1376, 1451, 1477])

    fhr_normal = fhr.drop(abnormal_curves)
    uc_normal = uc.drop(abnormal_curves)
    y_normal = y.drop(abnormal_curves)

    print(fhr_normal.shape, uc_normal.shape, y_normal.shape)

    fhr_nans = fhr_normal.copy()
    fhr_nans[(fhr_nans <= 0) | (fhr_nans >= 250)] = np.nan

    uc_nans = uc_normal.copy()
    uc_nans[uc_nans <= 0] = np.nan

    y_nans = y_normal.copy()

    print(fhr_nans.shape, uc_nans.shape, y_nans.shape)

    return fhr_nans, uc_nans, y_nans


def hist_box_plot(data):
    f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.85, .15)}, figsize=(10, 10))

    sns.boxplot(data, ax=ax_box)
    sns.histplot(data, ax=ax_hist, stat='percent')
    f.tight_layout()

    ax_box.set(xlabel='')
    plt.show()

def plot_signal(last_valid, first_valid, idx):
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    fhr_nans, uc_nans, y_nans = get_data()
    sys.stdout = save_stdout

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    color = 'red' if y_nans[idx] == 1 else 'blue'
    axes[0].plot(fhr_nans.loc[idx].index, fhr_nans.loc[idx].values, color=color)
    axes[1].plot(uc_nans.loc[idx].index, uc_nans.loc[idx].values, color=color)
    for ax in axes:
        ax.axvline(last_valid[idx], color='green')
        ax.axvline(first_valid[idx], color='red')
        ax.axvline(last_valid[idx] - 30 * 60, color='green')
        ax.set_xlim((uc_nans.columns.min(), uc_nans.columns.max()))
        ax.set_title(f'{idx}')
    fig.tight_layout()
    plt.show()

def one_test(idx):
    fhr_nans, uc_nans, y_nans = get_data()

    print(valid_index(uc_nans.loc[idx].copy(), kind='last'))


def global_test():
    fhr_nans, uc_nans, y_nans = get_data()

    first_valid_index_data = pd.DataFrame(columns=['FHR', 'UC'], index=fhr_nans.index)
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    first_valid_index_data['FHR'] = fhr_nans.apply(lambda x: valid_index(x, kind='first'), axis=1)
    first_valid_index_data['UC'] = uc_nans.apply(lambda x: valid_index(x, kind='first'), axis=1)
    first_valid_idxs = first_valid_index_data.max(axis=1)
    sys.stdout = save_stdout

    print(first_valid_idxs.describe())

    last_valid_index_data = pd.DataFrame(columns=['FHR', 'UC'], index=fhr_nans.index)
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    last_valid_index_data['FHR'] = fhr_nans.apply(lambda x: valid_index(x, kind='last'), axis=1)
    last_valid_index_data['UC'] = uc_nans.apply(lambda x: valid_index(x, kind='last'), axis=1)
    last_valid_idxs = last_valid_index_data.min(axis=1)
    sys.stdout = save_stdout

    print(last_valid_idxs.describe())

    lenght_data = (abs(last_valid_idxs - first_valid_idxs) / 60)
    hist_box_plot(lenght_data)

    print(lenght_data.describe())
    print(lenght_data.index[lenght_data <= 30])
    print(lenght_data.sort_values())

    return first_valid_idxs, last_valid_idxs

if __name__ == '__main__':
    first_valid_idxs, last_valid_idxs = global_test()
    plot_signal(last_valid_idxs, first_valid_idxs, 1482)
    # one_test(idx=1482)
