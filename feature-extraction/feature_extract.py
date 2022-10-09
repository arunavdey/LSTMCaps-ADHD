import nibabel as nib
import numpy as np
import os
import cv2

def reho(scan):
    pass

def fft(scan):
    pass

def falff(path):
    pass

def sbc(scan):
    pass


# TODO
# use c6mprage to remove the empty space around brain
def density(path):
    segments = list()

    gm_path = os.path.join(path, "anat", "c1mprage_skullstripped.nii")
    segments.append(nib.load(gm_path).get_fdata()) 
    wm_path = os.path.join(path, "anat", "c2mprage_skullstripped.nii")
    segments.append(nib.load(wm_path).get_fdata()) 
    csf_path = os.path.join(path, "anat", "c3mprage_skullstripped.nii")
    segments.append(nib.load(csf_path).get_fdata()) 

    res = list()

    for s in segments:
        dark = np.sum(s == 0)
        light = np.sum(s > 0)
        res.append(light/(dark+light))

    total = sum(res)
    ratio = 1 / total

    res[0] *= ratio
    res[1] *= ratio
    res[2] *= ratio

    return res
    

if __name__ == "__main__":
    print("Feature Extraction")

    home = "/home/arunav"
    path = os.path.join(home, "Assets", "test_data", "sub15213")

    gm, wm, csf = density(path)
    print(f"GM Density: {gm}")
    print(f"WM Density: {wm}")
    print(f"CSF Density: {csf}")
    print(f"Total: {gm+wm+csf}")
