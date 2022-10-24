import nibabel as nib
import pandas as pd
import numpy as np
import os
import time
# from densities import density


def generate(subject, root_path, save_path="./features/"):
    # TODO
    # ROIs and Densities for sMRI
    fmri_path = os.path.join(
        root_path, f"train/{subject}/func/sfnwmrda{subject}_session_1_rest_1.nii")
    falff_path = os.path.join(
        root_path, f"train/{subject}/falff_{subject}_session_1_rest_1.nii")
    reho_path = os.path.join(
        root_path, f"train/{subject}/reho_{subject}_session_1_rest_1.nii")
    pcc_path = os.path.join(
        root_path, f"train/{subject}/sbc/pcc_seed_correlation_z.nii")
    mpfc_path = os.path.join(
        root_path, f"train/{subject}/sbc/mpfc_seed_correlation_z.nii")
    lTPJ_path = os.path.join(
        root_path, f"train/{subject}/sbc/lTPJ_seed_correlation_z.nii")
    rTPJ_path = os.path.join(
        root_path, f"train/{subject}/sbc/rTPJ_seed_correlation_z.nii")

    fmri = nib.load(fmri_path).get_fdata()
    falff = nib.load(falff_path).get_fdata()
    reho = nib.load(reho_path).get_fdata()
    pcc = nib.load(pcc_path).get_fdata()
    mpfc = nib.load(mpfc_path).get_fdata()
    lTPJ = nib.load(lTPJ_path).get_fdata()
    rTPJ = nib.load(rTPJ_path).get_fdata()

    fmri_shape = fmri.shape

    features = ['x', 'y', 'z', 'falff', 'reho', 'pcc', 'mpfc', 'lTPJ', 'rTPJ']

    df = pd.DataFrame(columns = features)

    count = 0

    tot = fmri_shape[0] * fmri_shape[1] * fmri_shape[2]

    for x in range(fmri_shape[0]):
        for y in range(fmri_shape[1]):
            for z in range(fmri_shape[2]):


    # for x in range(25, 26):
    #     for y in range(25, 26):
    #         for z in range(25, 26):
                temp_df = list()
                temp_df.append(x)
                temp_df.append(y)
                temp_df.append(z)
                temp_df.append(falff[x][y][z])
                temp_df.append(reho[x][y][z])
                temp_df.append(pcc[x][y][z][0])
                temp_df.append(mpfc[x][y][z][0])
                temp_df.append(lTPJ[x][y][z][0])
                temp_df.append(rTPJ[x][y][z][0])
                df.loc[count] = temp_df
                count += 1
                print(f"\nProgress: {(count*100)/tot}")

    save = os.path.join(save_path, f"{subject}_features.csv")
    df.to_csv(save)

    print(f"generated features csv for subject {subject} at {save}")


if __name__ == "__main__":

    begin = time.time()
    
    print("Generating csv...")

    # subjects = [1018959, 1018959, 1019436, 1043241, 1266183, 1535233, 1541812, 1577042, 1594156, 1623716, 1638334, 1652369, 1686265, 1692275, 1735881, 1779922, 1842819, 1846346, 1873761, 1962503, 1988015, 1996183, 2014113, 2018106, 2026113, 2081148, 2104012, 2138826, 2299519, 2344857, 2360428, 2371032, 2554127, 2558999, 2572285, 2601925, 2618929, 2621228, 2640795, 2641332, 2703289, 2740232, 2768273, 2822304, 2903997, 2917777, 2930625, 3103809, 3119327, 3154996, 3160561, 3170319, 3310328, 3434578, 3486975, 3519022, 3611827, 3699991, 3713230, 3813783, 3884955, 3902469, 3912996, 3917422, 3972472, 3972956, 4104523, 4154182, 4275075, 4362730, 4601682, 5216908, 6346605, 6453038, 7129258, 7415617, 7774305, 8083695, 8263351, 8337695, 8432725, 8628223, 8658218, 9922944]
    subjects = [1018959]

    # root_path = "/home/arunav/Assets/ADHD200/kki_athena/"
    root_path = "/home/arunav/Assets/ADHD200/"

    for subject in subjects:
        generate(subject, root_path)
    
    end = time.time()

    print(f"Completed in:\t{end - begin}s")
