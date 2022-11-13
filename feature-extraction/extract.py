import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm

dir_home = os.path.join("/mnt", "d")
dir_adhd200 = os.path.join(dir_home, "Assets", "ADHD200")
sites = ["Peking_1", "Peking_2", "Peking_3"]
controlFunc = pd.DataFrame()
adhdFunc = pd.DataFrame()

'''
# figure out a way to use athena pipeline and niak together for these features

def smri(subject, dx, site):
    anat_path = os.path.join(
        dir_adhd200, f"{site}_niak","anat_kki", f"X_{subject}", f"anat_X_{subject}_classify_stereolin.nii.gz")
    anat = nib.load(anat_path).get_fdata()

    # features = ['x', 'y', 'z', 'class', 'gm_density', 'wm_density', 'dx', 'idx'] # 8
    features = ['x', 'y', 'z', 'class', 'dx']

    df = pd.DataFrame(columns=features)

    for x in range(anat.shape[0]):
        for y in range(anat.shape[1]):
            for z in range(anat.shape[2]):
                row = [x, y, z]
                row.append(anat[x][y][z])
                row.append(dx)
                df.loc[len(df.index)] = row

    print(f"Generated anatomical features for {subject} from {site}")

    return df
'''

def fmri(subject, dx, site):
    falff_path = os.path.join(dir_adhd200, f"{site}_athena", f"{site}_falff_filtfix",
                              f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")
    falff = nib.load(falff_path).get_fdata()  # 49, 58, 47

    reho_path = os.path.join(dir_adhd200, f"{site}_athena", f"{site}_reho_filtfix",
                             f"{subject}", f"reho_{subject}_session_1_rest_1.nii.gz")
    reho = nib.load(reho_path).get_fdata()  # 49, 58, 47

    fc_path = os.path.join(dir_adhd200, f"{site}_athena", f"{site}_preproc",
                           f"{subject}", f"fc_snwmrda{subject}_session_1_rest_1.nii.gz")
    fc = nib.load(fc_path).get_fdata()  # 49, 58, 47, 1, 10

    features = ['x', 'y', 'z', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6',
                'fc7', 'fc8', 'fc9', 'fc10', 'falff', 'reho', 'dx']  # 16

    df = pd.DataFrame(columns=features)

    for x in range(falff.shape[0]):
        for y in range(falff.shape[1]):
            for z in range(falff.shape[2]):
                f0 = fc[x][y][z][0]
                f1 = falff[x][y][z]
                f2 = reho[x][y][z]
                
                if f1 != 0 and f2 != 0 and sum(f0) != 0:
                    row = [x, y, z]
                    for i in range(10):
                        row.append(fc[x][y][z][0][i])
                    row.append(falff[x][y][z])
                    row.append(reho[x][y][z])
                    row.append(dx)
                    df.loc[len(df.index)] = row

                else:
                    continue

    print(f"Generated functional features for {subject} from {site}")

    return df

if __name__ == "__main__":

    for site in sites:
        pheno = pd.read_csv(os.path.join(dir_adhd200, f"{site}_athena", f"{site}_preproc", f"{site}_phenotypic.csv"))

        for ind in tqdm.tqdm(pheno.index):
            subID = pheno["ScanDir ID"][ind]
            dx = pheno["DX"][ind]

            print(f"Current subject: {subID}\nSite: {site}")
            if dx == 0:
                controlFunc = pd.concat([controlFunc, fmri(subID, dx, site)], axis=0)
            else:
                adhdFunc = pd.concat([adhdFunc, fmri(subID, dx, site)], axis=0)

        adhdFunc.to_csv(f"features/{site}_adhd_func.csv", index=False)
        controlFunc.to_csv(f"features/{site}_control_func.csv", index=False)
