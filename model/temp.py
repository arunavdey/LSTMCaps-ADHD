import os

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from tensorflow.keras.utils import to_categorical


def load_data():
    dir_home = os.path.join("/mnt", "d")
    dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "KKI_athena")

    pheno_path = os.path.join(
        dir_athena, "KKI_preproc", "KKI_phenotypic.csv")
    pheno = pd.read_csv(pheno_path)

    fmri_path = os.path.join(dir_athena, "KKI_preproc")

    subs = pheno["ScanDir ID"].to_numpy()

    x = list()

    for sub in tqdm.tqdm(subs, desc='Loading x'):
        scan_path = os.path.join(
            fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)
    y = np.transpose(pheno["DX"].to_numpy())

    for i in range(y.size):
        if y[i] > 0:
            y[i] = 1

    x_train, x_test = x[:40], x[40:50]
    y_train, y_test = y[:40], y[40:50]

    print(y_test)

    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    # return (x_train, y_train), (x_test, y_test)


load_data()
