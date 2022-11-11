import os

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from tensorflow.keras.utils import to_categorical


def load_data():
    dir_home = os.path.join("/mnt", "d")
    kki_athena = os.path.join(dir_home, "Assets", "ADHD200", "KKI_athena")
    peking1_athena = os.path.join(
        dir_home, "Assets", "ADHD200", "Peking_1_athena")

    kki_pheno_path = os.path.join(
        kki_athena, "KKI_preproc", "KKI_phenotypic.csv")
    kki_pheno = pd.read_csv(kki_pheno_path)

    peking1_pheno_path = os.path.join(
        peking1_athena, "Peking_1_preproc", "Peking_1_phenotypic.csv")
    peking1_pheno = pd.read_csv(peking1_pheno_path)

    kki_fmri_path = os.path.join(kki_athena, "KKI_preproc")
    peking1_fmri_path = os.path.join(peking1_athena, "Peking_1_preproc")

    kki_subs = kki_pheno["ScanDir ID"].to_numpy()
    peking1_subs = peking1_pheno["ScanDir ID"].to_numpy()

    x = list()

    for sub in tqdm.tqdm(kki_subs, desc='Loading kki'):
        scan_path = os.path.join(
            kki_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    for sub in tqdm.tqdm(peking1_subs, desc='Loading peking 1'):
        scan_path = os.path.join(
            peking1_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)
    y = kki_pheno["DX"].to_numpy()
    # print(y)
    y = np.append(y, peking1_pheno["DX"].to_numpy())

    # print(y)

    for i in range(y.size):
        if y[i] > 0:
            y[i] = 1

    print(x.shape)
    print(y.shape)

    x_train, x_test = x[:70], x[70:80]
    y_train, y_test = y[:70], y[70:80]

    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    # print(y_train)

    # return (x_train, y_train), (x_test, y_test)


load_data()
