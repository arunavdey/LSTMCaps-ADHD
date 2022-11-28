import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os

def load_data_csv(site):
    data = pd.read_csv(f"../feature-extraction/features/{site}_func.csv")

    x = data.iloc[1:, 0:-1]
    y = data.iloc[1:, -1]

    ss = StandardScaler()
    x = ss.fit_transform(x)

    x = x.astype('float32') / 255.0 # normalised to between 0 and 1
    y = y.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, shuffle = False)

    return (x_train, y_train), (x_test, y_test)

def load_data_mri(site):
    dir_home = os.path.join("/mnt", "hdd")
    athena = os.path.join(dir_home, "Assets", "ADHD200", f"{site}_athena")

    pheno_path = os.path.join(
        athena, f"{site}_preproc", f"{site}_phenotypic.csv")
    pheno = pd.read_csv(pheno_path)

    preproc = os.path.join(athena, f"{site}_preproc")

    subs = pheno["ScanDir ID"].to_numpy()

    x = list()

    for sub in subs:
        scan_path = os.path.join(
            preproc, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")

        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)

    y = pheno["DX"].to_numpy()


    x_train, x_test = x[:66], x[66:82]
    y_train, y_test = y[:66], y[66:82]

    # x_train = x_train.reshape(-1, 197, 233, 189, 1).astype('float32') / 255.
    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.

    # x_test = x_test.reshape(-1, 197, 233, 189, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.

    y_train = to_categorical(y_train.astype('float32'))

    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)
