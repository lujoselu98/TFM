"""
    Just to prove same file pickle and csv
"""
import pandas as pd

import Utils.paths as paths


def _integrity_checks() -> None:
    fhr = pd.read_csv(f"{paths.ORIGINAL_DATA_PATH}/fhr_ctu-chb.csv", index_col=0, compression='gzip')
    uc = pd.read_csv(f"{paths.ORIGINAL_DATA_PATH}/uc_ctu-chb.csv", index_col=0, compression='gzip')
    clinical = pd.read_csv(f"{paths.ORIGINAL_DATA_PATH}/clinical_ctu-chb.csv", index_col=0, compression='gzip')

    fhr_pickle = pd.read_pickle(f"{paths.ORIGINAL_DATA_PATH}/fhr_ctu-chb.pickle")
    uc_pickle = pd.read_pickle(f"{paths.ORIGINAL_DATA_PATH}/uc_ctu-chb.pickle")
    clinical_pickle = pd.read_pickle(f"{paths.ORIGINAL_DATA_PATH}/clinical_ctu-chb.pickle")

    print(f"{fhr_pickle.equals(fhr)=}")
    print(f"{uc_pickle.equals(uc)=}")
    print(f"{clinical_pickle.equals(clinical)=}")


if __name__ == '__main__':
    _integrity_checks()
