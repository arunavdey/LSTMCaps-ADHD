"""
Generates a CSV file for each subject with the features 
"""
import nibabel as nib
import pandas as pd
import numpy as np
import os

def generate(subject, root_path):
    pass


if __name__ == "__main__":
    print("Generate CSV")
    subject = 1018959
    # features = list()
    root_path = "/home/arunav/Assets/ADHD200/kki_athena/"
    falff_path = os.path.join(root_path, f"KKI_falff_filtfix/KKI/{subject}/falff_{subject}_session_1_rest_1.nii.gz")
    reho_path = os.path.join(root_path, f"KKI_reho_filtfix/KKI/{subject}/reho_{subject}_session_1_rest_1.nii.gz")

    falff = nib.load(falff_path).get_fdata()
    reho = nib.load(reho_path).get_fdata()

    falff_shape = falff.shape # tuple
    reho_shape = reho.shape

    df = pd.DataFrame(columns = ('x', 'y', 'z','falff', 'reho'))

    count = 0

    # for x in range(falff_shape[0]):
    #     for y in range(falff_shape[1]):
    #         for z in range(falff_shape[2]):
    for x in range(25, 30):
        for y in range(25, 30):
            for z in range(25, 30):
                temp_df = list()
                temp_df.append(x)
                temp_df.append(y)
                temp_df.append(z)
                temp_df.append(falff[x][y][z])
                temp_df.append(reho[x][y][z])
                print(temp_df)
                df.loc[count] = temp_df
                count += 1

    df.to_csv(f"{subject}_features.csv")


