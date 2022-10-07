import os
import nibabel as nib
import numpy as np


def loadData(path, verbose=False):
    if os.path.exists(path):
        print("Loaded {}".format(path))
        img_load = nib.load(path).get_fdata()
        if verbose:
            shape = img_load.shape
            print("Shape: {}".format(shape))
        return img_load


def getFilePaths(path, flag = 1, verbose=False):
    filePaths = list()
    if os.path.exists(path):
        for sub in os.listdir(path):
            d = os.path.join(path, sub)
            if os.path.isdir(d) and not d.endswith('Outputs'):
                anat, func = os.listdir(d)
                if (anat.endswith('func')):
                    anat, func = func, anat
                if flag == 2:
                    func = os.path.join(d, func)
                    for file in os.listdir(func):
                        if not os.path.isdir(file):
                            filePaths.append(os.path.join(func, file))
                if flag == 1:
                    anat = os.path.join(d, anat)
                    for file in os.listdir(anat):
                        if not os.path.isdir(file):
                            filePaths.append(os.path.join(anat, file))
    else:
        if verbose:
            print("Specified path doesn't exist: {}".format(path))
    return filePaths


def saveAsPNG(path, data, flag = 1, tag = ''):
    sh = data.shape
    paths = list()
    if os.path.exists(path):
        if flag == 1:
            for i in range(data.shape[1]):
                sl = data[:, i, :]
                if not np.any(sl):
                    continue
                fname = f'structural_{i}_{tag}.png'
                p = os.path.join(path, fname)
                matplotlib.image.imsave(p, sl, cmap='gray')
                paths.append(p)
        if flag == 2:
            #TODO
            print("No")
    return paths


def deleteDir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
