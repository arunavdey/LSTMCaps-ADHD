import nibabel as nib
import numpy as np
import os

def density(anat, gm, wm, csf):

    segments = [gm, wm, csf]
    res = list()

    emptyCount = np.sum(anat == 0)

    for s in segments:
        dark = np.sum(s == 0) - emptyCount
        light = np.sum(s > 0)
        res.append(light/(dark+light))

    return res
