import nibabel as nib
import os

def load_nii(path):
    return nib.load(path).get_fdata()


if __name__ == "__main__":
    subject = 1018959

    dir_home = os.path.join("/mnt", "d")
    dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "KKI_athena")
    dir_niak = os.path.join(dir_home, "Assets", "ADHD200", "KKI_niak")

    anat_path = os.path.join(dir_niak, "anat_kki", f"X_{subject}", f"anat_X_{subject}_classify_stereolin.nii.gz")
    falff_path = os.path.join(dir_athena, "KKI_falff_filtfix", f"{subject}", f"falff_{subject}_session_1_rest_1.nii.gz")

    anat = load_nii(anat_path)
    falff = load_nii(falff_path)

    print(anat.shape)
    print(falff.shape)
