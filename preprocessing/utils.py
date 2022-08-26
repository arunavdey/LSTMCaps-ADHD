import os
import nibabel as nib
import matplotlib
import numpy as np


def loadData(path, verbose=False):
    """
    Loads the .nii file and returns it as an object
    """
    if os.path.exists(path):
        print("Loaded {}".format(path))
        img_load = nib.load(path).get_fdata()
        if verbose:
            shape = img_load.shape
            print("Shape: {}".format(shape))
        return img_load


def getFilePaths(path, hospital, functional=True, structural=True, verbose=False):
    """
    Returns all the filepaths under the directory
    """
    rootDir = os.path.join(path, hospital)
    filePaths = list()
    if os.path.exists(rootDir):
        for sub in os.listdir(rootDir):
            d = os.path.join(rootDir, sub)
            if os.path.isdir(d):
                func, anat = os.listdir(d)
                if functional:
                    func = os.path.join(d, func)
                    for file in os.listdir(func):
                        if not os.path.isdir(file):
                            filePaths.append(os.path.join(func, file))
                if structural:
                    anat = os.path.join(d, anat)
                    for file in os.listdir(anat):
                        if not os.path.isdir(file):
                            filePaths.append(os.path.join(anat, file))
    else:
        if verbose:
            print("Specified path doesn't exist: {}".format(rootDir))
    return filePaths


def saveAsPNG(path, data, mri='structural'):
    """
    Saves each slice of a NumPy array as a PNG 
    """
    sh = data.shape
    paths = list()
    if os.path.exists(path):
        if mri == 'structural':
            for i in range(data.shape[1]):
                sl = data[:, i, :]
                if not np.any(sl):
                    continue
                fname = f'structural_{i}.png'
                p = os.path.join(path, fname)
                matplotlib.image.imsave(p, sl, cmap='gray')
                paths.append(p)
        if mri == 'functional':
            #TODO
            print("No")
    return paths


def deleteDir(path):
    """
    Deletes the specified directory
    """
    if os.path.exists(path):
        os.remove(path)
