"""
Generates a CSV file for each subject with the features 
"""
import nibabel as nib
import pandas as pd
import numpy as np
import os

def generate(subject, root_path, save_path = "./"):
    # TODO
    # add other features to generate code
    falff_path = os.path.join(root_path, f"KKI_falff_filtfix/KKI/{subject}/falff_{subject}_session_1_rest_1.nii.gz")
    reho_path = os.path.join(root_path, f"KKI_reho_filtfix/KKI/{subject}/reho_{subject}_session_1_rest_1.nii.gz")

    falff = nib.load(falff_path).get_fdata()
    reho = nib.load(reho_path).get_fdata()

    falff_shape = falff.shape # tuple
    reho_shape = reho.shape

    df = pd.DataFrame(columns = ('x', 'y', 'z','falff', 'reho'))

    count = 0

    for x in range(falff_shape[0]):
        for y in range(falff_shape[1]):
            for z in range(falff_shape[2]):
                temp_df = list()
                temp_df.append(x)
                temp_df.append(y)
                temp_df.append(z)
                temp_df.append(falff[x][y][z])
                temp_df.append(reho[x][y][z])
                print(temp_df)
                df.loc[count] = temp_df
                count += 1

    save = os.path.join(save_path, f"{subject}_features.csv")
    df.to_csv(save)

    print(f"Generated features csv for subject {subject} at {save}")


if __name__ == "__main__":
    print("Generating csv...")
    # TODO
    # add all subjects as a list

    subject = 1018959
    root_path = "/home/arunav/Assets/ADHD200/kki_athena/"

    generate(subject, root_path)
