import os
import nibabel as nib
import tensorflow as tf
from deepbrain import Extractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as sitk
import tqdm
import math
import shutil

from utils import loadData, saveAsPNG, deleteDir

tf.compat.v1.disable_v2_behavior()

class preprocess:
    def __init__(self, path, flag=1, savePath=None):
        self.path = path # path to .nii; 3D for sMRI and 4D for fMRI
        self.img = loadData(self.path) # 4D or 3D numpy array
        self.flag = flag

        splitpath = path.split('/')
        subj = splitpath[-3]
        opdir = os.path.join(savePath, subj)
        if not os.path.exists(opdir):
            os.mkdir(opdir)
        self.savePath = opdir


    def run(self):
        self.img = self.intensityNormalisation()
        self.img = self.skullStrip()
        # self.img = self.cropImage()
        self.img = self.addPadding()
        self.biasFieldCorrection()
        # self.tissueSegment()
        # self.showScan()


    def skullStrip(self):
        ext = Extractor()
        prob = ext.run(self.img)
        mask = prob > 0.5
        br_img = mask * self.img
        skull_img = self.img - br_img
        return br_img


    def intensityNormalisation(self):
        img = sitk.ReadImage(self.path)
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(255)
        rescaleFilter.SetOutputMinimum(0)
        image = rescaleFilter.Execute(img)
        op = os.path.join(self.savePath, 'normalised.nii')
        sitk.WriteImage(image, op)
        return loadData(op)


    def tissueSegment(self):
        paths = saveAsPNG(self.savePath, self.img, flag = 1)

        print(self.img.shape)

        for img_path in paths:
            image = cv2.imread(img_path)
            # b = image[:, :, np.newaxis]
            print(image.shape)

            # b,g,r = cv2.split(image)
            # b = cv2.equalizeHist(b)
            # g = cv2.equalizeHist(g)
            # r = cv2.equalizeHist(r)
            # equ = cv2.merge((b,g,r))
            # equ = cv2.cvtColor(equ,cv2.COLOR_BGR2RGB)

            # clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(16,16))
            # b,g,r = cv2.split(equ)
            # b = clahe.apply(b)
            # g = clahe.apply(g)
            # r = clahe.apply(r)
            # bgr = cv2.merge((b,g,r))
            # cl = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB, 0.5)
            # gray = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY, 0.5) 
            # (T, thresh) = cv2.threshold(gray, 180, 220, cv2.THRESH_BINARY)


        # return gray



    def biasFieldCorrection(self):
        img = sitk.ReadImage(self.path, imageIO="NiftiImageIO")
        n4 = sitk.N4BiasFieldCorrection(img)
        n4.inputs.dimension = 3
        n4.inputs.shrink_factor = 3
        n4.inputs.n_iterations = [20, 10, 10, 5]
        res = n4.run()
        plt.imshow(res)
        plt.show()
        # sitk.WriteImage(res, )


        # saveAsPNG(image, self.savePath)

    def cropImage(self):
        mask = self.img == 0
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        cropped = self.img[top_left[0]:bottom_right[0], :, top_left[1]:bottom_right[1]]

        return cropped

    def addPadding(self, height=256, width=256):
        h, _, w = self.img.shape
        final = np.zeros((height, self.img.shape[1], width))
        pad_left = int((width - w) // 2)
        pad_top = int((height - h) // 2)
        final[pad_top:pad_top + h, :, pad_left:pad_left + w] = self.img

        return final


    def edgeDetect(self):
        pass


    def showScan(self, verbose=False):
        # shape = self.img.shape
        # for i in range(0, shape[1], 4):
        #     img = self.img[:, i, :]
        #     # print(img.shape)
        #     plt.imshow(img, cmap = 'gray')
        #     plt.show()
        img = self.img[:, 50, :]
        # print(img.shape)
        plt.imshow(img, cmap = 'gray')
        plt.show()

        # if sl >= 0:
        #     if len(shape) > 3:
        #         img = self.img[:, :, sl, 0]
        #         plt.imshow(img, cmap='gray')
        #         plt.show()
        #         if verbose:
        #             print("Shape: {}".format(img.shape))
        #     else:
        #         img = self.img[:, sl, :]
        #         plt.imshow(img, cmap='gray')
        #         plt.show()
        #         if verbose:
        #             print("Shape: {}".format(img.shape))
        # else:
        #     for s2 in range(100, shape[2]):
        #         if len(shape) > 3:
        #             img = self.img[:, :, s2, 0]
        #             plt.imshow(img, cmap='gray')
        #             plt.show()
        #             if verbose:
        #                 print("Shape: {}".format(img.shape))
        #             # for s3 in range(shape[3]):
        #             #     img = data[:, :, s2, s2]
        #             #     plt.imshow(img)
        #             #     plt.show()
        #         else:
        #             img = self.img[:, :, s2]
        #             plt.imshow(img, cmap='gray')
        #             plt.show()
        #             if verbose:
        #                 print("Shape: {}".format(img.shape))

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
