import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import os

def load_data_csv(site):
    data = pd.read_csv(f"../feature-extraction/features/{site}_func.csv")

    x = data.iloc[1:, 0:-1]
    y = data.iloc[1:, -1]

    classCount = [0, 0, 0, 0]

    for Y in y:
        classCount[int(Y)] += 1

    print(classCount)

    ss = StandardScaler()
    x = ss.fit_transform(x)

    x = x.astype('float32') / 255.0
    y = y.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle = False)

    # y_train = to_categorical(y_train.astype('float32'))
    # y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def load_data_mri(site):
    dir_home = os.path.join("/mnt", "hdd")
    x, y = list(), list()

    print(f"Loading data for site {site}\n")
    athena = os.path.join(dir_home, "Assets", "ADHD200", f"{site}_athena")

    pheno_path = os.path.join(
        athena, f"{site}_preproc", f"{site}_phenotypic.csv")
    pheno = pd.read_csv(pheno_path)

    preproc = os.path.join(athena, f"{site}_preproc")

    subs = pheno["ScanDir ID"].to_numpy()
    dx = pheno["DX"].to_numpy()

    for ind in pheno.index:
        scan_path = os.path.join(
            preproc, f"{subs[ind]}", f"snwmrda{subs[ind]}_session_1_rest_1.nii.gz")
            # preproc, f"{subs[ind]}", f"falff_{subs[ind]}_session_1_rest_1.nii.gz")
            # preproc, f"{subs[ind]}", f"reho_{subs[ind]}_session_1_rest_1.nii.gz")
            # preproc, f"{subs[ind]}", f"fc_snwmrda{subs[ind]}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()

        for i in range(16):
            x.append(scan[:, :, :, i])
            y.append(dx[ind])

    x = np.array(x)
    y = np.array(y)
    
    y = to_categorical(y.astype('float32'))

    x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size = 1064, test_size = 264, random_state=42, shuffle = False)
            # x, y, train_size = 118, test_size = 24, random_state=42, shuffle = False)

    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.

    # y_train = to_categorical(y_train.astype('float32'))
    # y_test = to_categorical(y_test.astype('float32'))

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)
