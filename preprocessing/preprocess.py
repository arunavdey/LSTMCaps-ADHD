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
        self.img = loadData(self.path, verbose=True) # 4D or 3D numpy array
        self.flag = flag

        splitpath = path.split('/')
        subj = splitpath[-3]
        opdir = os.path.join(savePath, subj)
        if os.path.exists(opdir):
            shutil.rmtree(opdir)
        os.mkdir(opdir)

        self.savePath = opdir


    def run(self):
        # print("Running Intensity Normalisation\n")
        # self.intensityNormalisation()
        print("Running Skull Stripping\n")
        self.img = self.skullStrip()
        print("Running Cropping\n")
        self.img = self.cropImage()
        print("Running Add Padding\n")
        self.img = self.addPadding()
        print(self.img.shape)

        img_paths = saveAsPNG(self.savePath, self.img, flag = 1, tag = 'ss_c_pad')

        # for ip in img_paths:
        #     print(ip)
            # ts = self.tissueSegment(ip) # 2D image
            # for t in range(ts.shape[1]):
            #     saveAsPNG(self.savePath, ts[:, t, :], flag = 1)
            

        # deleteDir(self.savePath)

    def edgeDetect(self):
        pass


    def tissueSegment(self, img_path):
        image = cv2.imread(img_path)
        # print(img_path)

        b,g,r = cv2.split(image)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        equ = cv2.merge((b,g,r))
        equ = cv2.cvtColor(equ,cv2.COLOR_BGR2RGB)

        clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(16,16))
        b,g,r = cv2.split(equ)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        bgr = cv2.merge((b,g,r))
        cl = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB, 0.5)

        gray = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY, 0.5) 
        return gray


    def skullStrip(self):
        ext = Extractor()
        prob = ext.run(self.img)
        mask = prob > 0.5
        br_img = mask * self.img
        skull_img = self.img - br_img
        return br_img

    def biasFieldCorrection(self):
        img = sitk.ReadImage(self.path, imageIO="NiftiImageIO")
        n4 = sitk.N4BiasFieldCorrection(img)
        n4.inputs.dimension = 3
        n4.inputs.shrink_factor = 3
        n4.inputs.n_iterations = [20, 10, 10, 5]
        res = n4.run()
        sitk.WriteImage(res, '/home/arunav/Assets/outputs/bf_out.nii')

    def intensityNormalisation(self):
        img = sitk.ReadImage(self.path)
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(255)
        rescaleFilter.SetOutputMinimum(0)
        image = rescaleFilter.Execute(img)
        sitk.WriteImage(image, os.path.join(self.savePath, 'normalised.nii'))

        # saveAsPNG(image, self.savePath)

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

    def showScan(self, sl, verbose=False):
        shape = self.img.shape
        if sl >= 0:
            if len(shape) > 3:
                img = self.img[:, :, sl, 0]
                plt.imshow(img, cmap='gray')
                plt.show()
                if verbose:
                    print("Shape: {}".format(img.shape))
            else:
                img = self.img[:, sl, :]
                plt.imshow(img, cmap='gray')
                plt.show()
                if verbose:
                    print("Shape: {}".format(img.shape))
        else:
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
