import nibabel as nib
import numpy as np
import os

def density(path):
    segments = list()

    gm_path = os.path.join(path, "c1anat_X_1018959_classify_stereolin.nii")
    segments.append(nib.load(gm_path).get_fdata()) 
    wm_path = os.path.join(path, "c2anat_X_1018959_classify_stereolin.nii")
    segments.append(nib.load(wm_path).get_fdata()) 
    csf_path = os.path.join(path, "c3anat_X_1018959_classify_stereolin.nii")
    segments.append(nib.load(csf_path).get_fdata()) 
    empty_path = os.path.join(path, "c6anat_X_1018959_classify_stereolin.nii")
    segments.append(nib.load(empty_path).get_fdata())

    res = list()

    emptyCount = np.sum(segments[3] > 0)

    for s in segments:
        dark = np.sum(s == 0) - emptyCount
        light = np.sum(s > 0)
        res.append(light/(dark+light))

    # total = sum(res)
    # ratio = 1 / total

    # res[0] *= ratio
    # res[1] *= ratio
    # res[2] *= ratio

    return res
    

if __name__ == "__main__":
    print("Densities")
    home = "/home/arunav"
    path = os.path.join(home, "Assets", "test_data", "segmented_1018959_anat")

    gm, wm, csf, _ = density(path)
    print(f"GM Density: {gm}")
    print(f"WM Density: {wm}")
    print(f"CSF Density: {csf}")
    print(f"Total: {gm+wm+csf}")
