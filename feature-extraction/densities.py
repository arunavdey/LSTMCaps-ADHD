import nibabel as nib
import numpy as np
import os

def density(path, sub):
    segments = list()

    gm_path = os.path.join(path, f"c1anat_X_{sub}_classify_stereolin.nii")
    segments.append(nib.load(gm_path).get_fdata()) 
    wm_path = os.path.join(path, f"c2anat_X_{sub}_classify_stereolin.nii")
    segments.append(nib.load(wm_path).get_fdata()) 
    csf_path = os.path.join(path, f"c3anat_X_{sub}_classify_stereolin.nii")
    segments.append(nib.load(csf_path).get_fdata()) 
    empty_path = os.path.join(path, f"c6anat_X_{sub}_classify_stereolin.nii")
    segments.append(nib.load(empty_path).get_fdata())

    res = list()

    emptyCount = np.sum(segments[3] > 0)

    for s in segments:
        dark = np.sum(s == 0) - emptyCount
        light = np.sum(s > 0)
        res.append(light/(dark+light))

    return res
    

if __name__ == "__main__":
    print("Densities")
    home = "/home/arunav"
    sub = "1018959"
    path = os.path.join(home, "Assets", "test_data", f"segmented_{sub}_anat")

    gm, wm, csf, _ = density(path, sub)
    print(f"GM Density: {gm}")
    print(f"WM Density: {wm}")
    print(f"CSF Density: {csf}")
    print(f"Total: {gm+wm+csf}")
