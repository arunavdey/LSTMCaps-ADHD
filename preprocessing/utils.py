# import json
import os
import nibabel as nib
import tensorflow as tf
from deepbrain import Extractor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import tqdm
# import math


def loadData(path, verbose=False):
    if os.path.exists(path):
        print("Loaded {}".format(path))
        img_load = nib.load(path).get_fdata()
        if verbose:
            shape = img_load.shape
            print("Shape: {}".format(shape))
        return img_load


def getFilePaths(path, hospital, functional=True, structural=True, verbose=False):
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


class preprocess:
    def __init__(self, path):
        self.path = path
        self.img = loadData(self.path)

    # def skullStrip(self, img):
    #     ext = Extractor()
    #     prob = ext.run(img)
    #     mask = prob > 0.5
    #     br_img = mask * img
    #     # skull_img = img - br_img
    #     return br_img

    def showScan(self, verbose=False):
        shape = self.img.shape
        for s2 in range(100, shape[2]):
            if len(shape) > 3:
                img = self.img[:, :, s2, 0]
                plt.imshow(img, cmap='gray')
                plt.show()
                if verbose:
                    print("Shape: {}".format(img.shape))
                # for s3 in range(shape[3]):
                #     img = data[:, :, s2, s2]
                #     plt.imshow(img)
                #     plt.show()
            else:
                img = self.img[:, :, s2]
                plt.imshow(img, cmap='gray')
                plt.show()
                if verbose:
                    print("Shape: {}".format(img.shape))

    def biasFieldCorrection(self):
        img = sitk.ReadImage(self.path, imageIO="NiftiImageIO")
        n4 = sitk.N4BiasFieldCorrection(img)
        n4.inputs.dimension = 3
        n4.inputs.shrink_factor = 3
        n4.inputs.n_iterations = [20, 10, 10, 5]
        res = n4.run()
        sitk.WriteImage(res, '/home/arunav/Assets/outputs/bf_out.nii')

    def intensityNormalisation(self):
        img = sitk.ReadImage(self.path, imageIO="NiftiImageIO")
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(255)
        rescaleFilter.SetOutputMinimum(0)
        image = rescaleFilter.Execute(img)
        sitk.WriteImage(image, '/home/arunav/Assets/outputs/in_out.nii')

    def cropImage(self):
        mask = self.img == 0
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        cropped = self.img[top_left[0]:bottom_right[0],
                           top_left[1]:bottom_right[1], :]

        return cropped

    def addPadding(self, height=256, width=256):
        h, w, _ = self.img.shape
        final = np.zeros((height, width, self.img.shape[2]))
        pad_left = int((width - w) // 2)
        pad_top = int((height - h) // 2)
        final[pad_top:pad_top + h, pad_left:pad_left + w, :] = self.img

        return final

    # TODO

    # def tiltCorrection(img):
    #     img = np.uint8(img[:, :, 120])
    #     contours = cv2.findContours(
    #         img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     print(contours)
        # plt.imshow(contours)
        # plt.show()
        # mask = np.zeros(img.shape, np.uint8)

        # c = max(contours, key=cv2.contourArea)
        # (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        # cv2.ellipse(img, ((x, y), (MA, ma), angle), color=(0, 255, 0), thickness=2)
        # rmajor = max(MA, ma)/2

        # if angle > 90:
        #     angle -= 90
        # else:
        #     angle += 96

        # xtop = x + math.cos(math.radians(angle))*rmajor
        # ytop = y + math.sin(math.radians(angle))*rmajor
        # xbot = x + math.cos(math.radians(angle+180))*rmajor
        # ybot = y + math.sin(math.radians(angle+180))*rmajor
        # cv2.line(img, (int(xtop), int(ytop)),
        #          (int(xbot), int(ybot)), (0, 255, 0), 3)
        # plt.imshow(img)
        # plt.show()
        # M = cv2.getRotationMatrix2D((x, y), angle-90, 1)
        # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        # plt.imshow(img)
        # plt.show()
