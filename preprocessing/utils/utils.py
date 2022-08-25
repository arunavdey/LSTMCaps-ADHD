import json
import os
import nibabel as nib

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from deepbrain import Extractor
import numpy as np
import matplotlib.pyplot as plt

def loadData(path):
    if os.path.exists(path):
        print("Loaded {}".format(path))
        img_load = nib.load(path).get_fdata()
        shape = img_load.shape
        return img_load


def getFilePaths(path, hospital, functional = True, structural = True, verbose = False):
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


def showScan(data):
    shape = data.shape
    print(shape)
    print(len(data))
    for s2 in range(shape[2]/2, shape[2]):
        if len(shape) > 3:
            img = data[:, :, s2, 0]
            plt.imshow(img)
            plt.show()
            # for s3 in range(shape[3]):
            #     img = data[:, :, s2, s2]
            #     plt.imshow(img)
            #     plt.show()
        else:
            img = data[:, :, s2]
            plt.imshow(img)
            plt.show()


def skullStrip(data):
    ext = Extractor()
    prob = ext.run(data)
    mask = prob > 0.5
    br_img = mask * data
    skull_img = data - br_img
    for i in range (data.shape[0]):
        fig, ax = plt.subplt(1, 2, figsize = (10, 10))
        img1 = data[i, :, :]
        img2 = br_img[i, :, :]
        plt.imshow(img1)
        plt.imshow(img2)
        plt.show()
