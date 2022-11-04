import nibabel as nib
import numpy as np
import pandas as pd
import os

def run(subject):
    dir_home = os.path.expanduser("~")
    # dir_data = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena", "KKI_preproc", "KKI", f"{subject}")
    dir_data = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena")

    # fractional amplitude of low frequency fluctuations
    falff_path = os.path.join(dir_data, "KKI_falff_filtfix", "KKI", f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")
    falff = nib.load(falff_path).get_fdata()

    # regional homogenity
    reho_path = os.path.join(dir_data, "KKI_reho_filtfix", "KKI", f"{subject}", f"reho_{subject}_session_1_rest_1.nii.gz")
    reho = nib.load(reho_path).get_fdata()

    # functional connectivity
    fc_path = os.path.join(dir_data, "KKI_preproc", "KKI", f"{subject}", f"fc_snwmrda{subject}_session_1_rest_1.nii.gz")
    fc = nib.load(fc_path).get_fdata()

    # mean of time series
    mean_path = os.path.join(dir_data, "KKI_preproc", "KKI", f"{subject}", f"wmean_mrda{subject}_session_1_rest_1.nii.gz")
    mean = nib.load(mean_path).get_fdata()

    print(falff.shape)  # 49, 58, 47
    print(reho.shape)   # 49, 58, 47
    print(fc.shape)     # 49, 58, 47, 1, 10
    print(mean.shape)   # 49, 58, 47

if __name__ == "__main__":
    subjects = [9922944]

    for sub in subjects:
        run(sub)
