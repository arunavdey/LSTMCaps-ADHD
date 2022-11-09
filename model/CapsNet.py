import tensorflow as tf
import pandas as pd
import numpy as np
import nibabel as nib
from tf.keras.layers import Input, Conv3D, Dense

dir_home = os.path.expanduser("~")
dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena")
dir_niak = os.path.join(dir_home, "Assets", "ADHD200", "kki_niak")


falff_path = os.path.join(dir_athena, "KKI_falff_filtfix", "KKI", f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")

falff = nib.load(falff_path).get_fdata()

# inputs = Input(shape = (falff.shape))
