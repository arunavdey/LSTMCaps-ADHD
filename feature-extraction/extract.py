import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm

from densities import density

dir_home = os.path.join("/mnt", "d")
dir_adhd200 = os.path.join(dir_home, "Assets", "ADHD200")
sites = ["KKI", "Peking_1", "Peking_2", "Peking_3"]
controlFunc = pd.DataFrame()
adhdFunc = pd.DataFrame()
controlAnat = pd.DataFrame()
adhdAnat = pd.DataFrame()

# figure out a way to use athena pipeline and niak together for these features

def smri(subject, dx, site):
    anat_path = os.path.join(dir_adhd200, f"{site}_athena",f"{site}_preproc", f"{subject}", f"wssd{subject}_session_1_anat.nii.gz")
    gm_path = os.path.join(dir_adhd200, f"{site}_athena",f"{site}_preproc", f"{subject}", f"wssd{subject}_session_1_anat_gm.nii.gz")
    wm_path = os.path.join(dir_adhd200, f"{site}_athena",f"{site}_preproc", f"{subject}", f"wssd{subject}_session_1_anat_wm.nii.gz")
    csf_path = os.path.join(dir_adhd200, f"{site}_athena",f"{site}_preproc", f"{subject}", f"wssd{subject}_session_1_anat_csf.nii.gz")

    anat = nib.load(anat_path).get_fdata()
    gm = nib.load(gm_path).get_fdata()
    wm = nib.load(wm_path).get_fdata()
    csf = nib.load(csf_path).get_fdata()

    features = ['x', 'y', 'z', 'gm_density', 'wm_density', 'csf_density', 'gm', 'wm', 'csf', 'dx'] # 10

    gm_density, wm_density, csf_density = density(anat, gm, wm, csf)

    df = pd.DataFrame(columns=features)

    for x in range(anat.shape[0]):
        for y in range(anat.shape[1]):
            for z in range(anat.shape[2]):
                f0 = gm[x][y][z]
                f1 = wm[x][y][z]
                f2 = csf[x][y][z]

                if f0 != 0 and f1 != 0 and f2 != 0:
                    row = [x, y, z, gm_density, wm_density, csf_density]
                    row.append(gm)
                    row.append(wm)
                    row.append(csf)
                    row.append(dx)
                    row.loc[len(df.index)] = row

                else:
                    continue

    print(f"Generated anatomical features for {subject} from {site}")

    return df

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
                controlAnat = pd.concat([controlAnat, smri(subID, dx, site)], axis=0)
            else:
                adhdFunc = pd.concat([adhdFunc, fmri(subID, dx, site)], axis=0)
                adhdAnat = pd.concat([adhdAnat, smri(subID, dx, site)], axis=0)

        adhdFunc.to_csv(f"features/{site}_adhd_func.csv", index=False)
        controlFunc.to_csv(f"features/{site}_control_func.csv", index=False)
