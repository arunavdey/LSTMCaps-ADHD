import nibabel as nib
import numpy as np
import pandas as pd
import os

from densities import density


dir_home = os.path.expanduser("~")
dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena")
# dir_niak = os.path.join(dir_home, "Assets", "ADHD200", "kki_niak")


# def smri(subject, dx, idx, save_path="./features/"):
#     # wm_path
#     # gm_path
#     # csf_path
#     # empty_path

#     features = ['x', 'y', 'z', 'gm', 'wm', 'csf', 'gm_density', 'wm_density', 'dx', 'idx']

def fmri(subject, dx, idx, save_path="./features/"):
    falff_path = os.path.join(dir_athena, "KKI_falff_filtfix", "KKI",
                              f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")
    falff = nib.load(falff_path).get_fdata()  # 49, 58, 47

    reho_path = os.path.join(dir_athena, "KKI_reho_filtfix", "KKI",
                             f"{subject}", f"reho_{subject}_session_1_rest_1.nii.gz")
    reho = nib.load(reho_path).get_fdata()  # 49, 58, 47

    fc_path = os.path.join(dir_athena, "KKI_preproc", "KKI",
                           f"{subject}", f"fc_snwmrda{subject}_session_1_rest_1.nii.gz")
    fc = nib.load(fc_path).get_fdata()  # 49, 58, 47, 1, 10

    features = ['x', 'y', 'z', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6',
                'fc7', 'fc8', 'fc9', 'fc10', 'falff', 'reho', 'dx', 'idx']

    df = pd.DataFrame(columns=features)

    for x in range(49):
        for y in range(58):
            for z in range(47):
                row = [x, y, z]
                for i in range(10):
                    row.append(fc[x][y][z][0][i])
                row.append(falff[x][y][z])
                row.append(reho[x][y][z])
                row.append(dx)
                row.append(idx)
                df.loc[len(df.index)] = row

    save = os.path.join(save_path, f"{subject}_features_func.csv")
    df.to_csv(save)

    print(f"Generated functional features for {subject} at {save}")


if __name__ == "__main__":
    pheno_path = os.path.join(
        dir_athena, "KKI_preproc", "KKI", "KKI_phenotypic.csv")
    pheno = pd.read_csv(pheno_path)

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

    for subID, dx, idx in subAll:
        fmri(subID, dx, idx)
