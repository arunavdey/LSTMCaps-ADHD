import nibabel as nib
import numpy as np
import pandas as pd
import os

dir_home = os.path.expanduser("~")
dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena")
# dir_niak = os.path.join(dir_home, "Assets", "ADHD200", "kki_niak")


def fmri(subject, dx, idx, save_path="./features/"):
    # fractional amplitude of low frequency fluctuations
    falff_path = os.path.join(dir_athena, "KKI_falff_filtfix", "KKI",
                              f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")
    falff = nib.load(falff_path).get_fdata()  # 49, 58, 47

    # regional homogenity
    reho_path = os.path.join(dir_athena, "KKI_reho_filtfix", "KKI",
                             f"{subject}", f"reho_{subject}_session_1_rest_1.nii.gz")
    reho = nib.load(reho_path).get_fdata()  # 49, 58, 47

    # functional connectivity
    fc_path = os.path.join(dir_athena, "KKI_preproc", "KKI",
                           f"{subject}", f"fc_snwmrda{subject}_session_1_rest_1.nii.gz")
    fc = nib.load(fc_path).get_fdata()  # 49, 58, 47, 1, 10

    # mean of time series
    mean_path = os.path.join(dir_athena, "KKI_preproc", "KKI",
                             f"{subject}", f"wmean_mrda{subject}_session_1_rest_1.nii.gz")
    mean = nib.load(mean_path).get_fdata()  # 49, 58, 47

    features = ['x', 'y', 'z', 'falff', 'reho', 'fc', 'dx', 'idx']

    df = pd.DataFrame(columns=features)

    tot = 49 * 48 * 47

    count = 0

    for x in range(25, 20):
        for y in range(25, 30):
            for z in range(25, 30):
                temp_df = [x, y, z]
                temp_df.append(falff[x][y][z])
                temp_df.append(reho[x][y][z])
                temp_df.append(fc[x][y][z][0])
                temp_df.append(dx)
                temp_df.append(idx)
                df.loc[count] = temp_df
                count += 1

    save = os.path.join(save_path, f"{subject}_features_func.csv")
    df.to_csv(save)

    print(f"Generated fMRI features for {subject} at {save}")


def smri(subject):
    pass


if __name__ == "__main__":
    pheno_path = os.path.join(
        dir_athena, "KKI_preproc", "KKI", "KKI_phenotypic.csv")
    pheno = pd.read_csv(pheno_path)
    # columns = pheno.columns

    """
    'ScanDir ID', 'Site', 'Gender', 'Age', 'Handedness', 'DX', 'Secondary Dx ', 'ADHD Measure', 'ADHD Index', 'Inattentive', 
    'Hyper/Impulsive', 'IQ Measure', 'Verbal IQ', 'Performance IQ', 'Full2 IQ', 'Full4 IQ', 'Med Status', 'QC_Rest_1', 
    'QC_Rest_2', 'QC_Rest_3', 'QC_Rest_4', 'QC_Anatomical_1', 'QC_Anatomical_2'
    """

    subAll = list()
    subADHD = list()
    subControl = list()

    for ind in pheno.index:
        subID = pheno["ScanDir ID"][ind]
        dx = pheno["DX"][ind]
        index = pheno["ADHD Index"][ind]

        subAll.append((subID, dx, index))

        if dx == 0:
            subControl.append((subID, dx, index))
        else:
            subADHD.append((subID, dx, index))

    for sub in subADHD:
        fmri(sub[0], sub[1], sub[2])
