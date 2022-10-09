import numpy as np
import time
import zipfile
import cv2
import os
import random
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import filedialog
from nibabel.testing import data_path
import nibabel as nib
from skimage import feature
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF

# filename = filedialog.askopenfilename(title='open')
home = "/home/arunav"
filename = os.path.join(home, "Assets", "ADHD-200", "PaloAlto", "sub04856", "anat", "mprage_anonymized.nii.gz")

res=1

img = nib.load(filename).get_fdata()
##x=50
##for i in range(5):
##    plt.subplot(5, 5,i + 1)
##    xc=img[:,:,x + i]
##    x=x+50
##    print(x)
##    plt.imshow(xc)
##    plt.gcf().set_size_inches(10, 10)
##plt.show()


t1 = img

print('t1.shape (%d, %d, %d)' % t1.shape)

fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
img_ax = np.rot90(t1[:, :, 180])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('sMRI axial')
plt.show()


fig = plt.figure()
b = fig.add_subplot(1, 1, 1)
img_ax1 = np.rot90(t1[120, :,:])
imgplot = plt.imshow(img_ax1 , cmap="gray")
b.axis('off')
b.set_title('sMRI sagital')

plt.show()



filename = os.path.join(home, "Assets", "ADHD-200", "PaloAlto", "sub04856", "func", "rest.nii.gz")
img = nib.load(filename).get_fdata()

t2 = img

fig = plt.figure()
b = fig.add_subplot(1, 1, 1)
img_cor = np.rot90(t2[:, :, 20,120])
imgplot = plt.imshow(img_cor, cmap="gray")
b.axis('off')
b.set_title('fMRI axial')

plt.show()



fig = plt.figure()
b = fig.add_subplot(1, 1, 1)
img_cor1 = np.rot90(t2[30, :, :,120])
imgplot = plt.imshow(img_cor1, cmap="gray")
b.axis('off')
b.set_title('fMRI sagital')

plt.show()


# def coregistration(img1,im2):
#     height, width = img2.shape
#     orb_detector = cv2.ORB_create(5000)
#     img11=img1.astype('uint8')
#     img22=img2.astype('uint8')
#     kp1, d1 = orb_detector.detectAndCompute(img11, None)
#     kp2, d2 = orb_detector.detectAndCompute(img22, None)
#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
#     matches = matcher.match(d1, d2)
#     matches = matches[:int(len(matches)*0.9)]
#     no_of_matches = len(matches)

#     out=(img1+img2)/2

#     return out

#     # p1 = np.zeros((no_of_matches, 2))
#     # p2 = np.zeros((no_of_matches, 2))
#     # for i in range(len(matches)):
#     #  p1[i, :] = kp1[matches[i].queryIdx].pt
#     #  p2[i, :] = kp2[matches[i].trainIdx].pt
#     # homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
#     # transformed_img = cv2.warpPerspective(img11,homography, (width, height))
    

# img1 = cv2.resize(img_ax1,(256,256))
# img2 = cv2.resize(img_ax1,(256,256))
# smri=coregistration(img1,img2)

# img1 = cv2.resize(img_cor,(256,256))
# img2 = cv2.resize(img_cor1,(256,256))
# fmri=coregistration(img1,img2)


# def normalisation(img):
#         resultimage = np.zeros((256,256))
#         final = normalizedimage = cv2.normalize(img,resultimage, 0, 100, cv2.NORM_MINMAX)
#         return final
    
# def filter1(img):
#         kernel = np.ones((5,5),np.float32)/25
#         im2 = cv2.filter2D(img,-1,kernel)
#         return im2


# fmri=normalisation(fmri)
# smri=normalisation(smri)
# fmri=filter1(fmri)
# smri=filter1(smri)

# fig = plt.figure()
# b = fig.add_subplot(1, 1, 1)
# imgplot = plt.imshow(smri, cmap="gray")
# b.axis('off')
# b.set_title('sMRI')

# plt.show()



# fig = plt.figure()
# b = fig.add_subplot(1, 1, 1)
# imgplot = plt.imshow(fmri, cmap="gray")
# b.axis('off')
# b.set_title('fMRI')

# plt.show()


nclass = 3
beta = 0.1
t0 = time.time()
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE1 = hmrf.classify(smri.reshape(smri.shape[0],smri.shape[1],1), nclass, beta)


fig = plt.figure()
a = fig.add_subplot(1, 3, 1)
img_ax = np.rot90(PVE1[:,:,:,0].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('sMRI CSF')
a = fig.add_subplot(1, 3, 2)
img_cor = np.rot90(PVE1[:,:,:,1].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('sMRI Gray Matter')
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(PVE1[:,:,:,2].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('sMRI White Matter')
plt.savefig('probabilities.png', bbox_inches='tight', pad_inches=0)
plt.show()



xc=PVE1[:,:,:,0].reshape(PVE1.shape[0],PVE1.shape[1])
sxc1 = xc#.astype("uint8")
xc=PVE1[:,:,:,1].reshape(PVE1.shape[0],PVE1.shape[1])
sxc2 = xc#.astype("uint8")
xc=PVE1[:,:,:,2].reshape(PVE1.shape[0],PVE1.shape[1])
sxc3 = xc#.astype("uint8")


nclass = 3
beta = 0.1
import time
t0 = time.time()
xc=fmri
xc=xc.reshape(fmri.shape[1],fmri.shape[0],1)
hmrf = TissueClassifierHMRF()
initial_segmentation1, final_segmentation1, PVE1 = hmrf.classify(xc, nclass, beta)


fig = plt.figure()
a = fig.add_subplot(1, 3, 1)
img_ax = np.rot90(PVE1[:,:,:,0].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('fMRI CSF')
a = fig.add_subplot(1, 3, 2)
img_cor = np.rot90(PVE1[:,:,:,1].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('fMRI Gray Matter')
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(PVE1[:,:,:,2].reshape(PVE1.shape[0],PVE1.shape[1],1))
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('fMRI White Matter')
plt.savefig('probabilities.png', bbox_inches='tight', pad_inches=0)
plt.show()


xc=PVE1[:,:,:,0].reshape(PVE1.shape[0],PVE1.shape[1])
fxc1 = xc#.astype("uint8")
xc=PVE1[:,:,:,1].reshape(PVE1.shape[0],PVE1.shape[1])
fxc2 = xc#.astype("uint8")
xc=PVE1[:,:,:,2].reshape(PVE1.shape[0],PVE1.shape[1])
fxc3 = xc#.astype("uint8")


def ReHo(xc):
        from sklearn.metrics.cluster import homogeneity_score
        import cv2
        import numpy as np
        img1_color =xc
        randomlist = []
        for i in range(256*256):
                n = random.randint(1,30)
                randomlist.append(n)
        hscore = homogeneity_score(xc.reshape(256*256), randomlist)
        print("Regional homogeneity is = " , hscore)
        return hscore

def fALFF(xc):
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        img1_color = xc
        t0 = 0
        t1 = 20
        n_samples = 1000
        xs = np.linspace(t0, t1, n_samples)
        ys = 7*np.sin(15 * 2 * np.pi * xs) + 3*np.sin(13 * 2 * np.pi * xs)
        np_fft = np.fft.fft(img1_color)
        frequencies = 1/n_samples * np.abs(np_fft) 
        #frequencies = np.fft.fftfreq(amplitudes) * amplitudes * 1/(t1-t0)
        print("falff is =",np.mean(frequencies))
        return np.mean(frequencies)

def seed(xc):
        import cv2
        image = xc*255
        gray_image = image.astype('uint8')#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_image], [0],
                         None, [256], [0, 10])
##        image = xc.astype('uint8')#cv2.imread('dataset')
##        gray_image1 = xc#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##        histogram1 = cv2.calcHist([gray_image1], [0],
##                          None, [256], [0, 256])
        print("seed histogram")
        print(np.mean(histogram))
        return np.mean(histogram)

        
def density(xc):
        import cv2
        import numpy as np
        img=xc*255#cv2.imread('dataset')
        img = img.astype('uint8')#cv2.
        #img = xc#cv2.imread()
        ht, wd = img.shape[:2]
        gray = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        veins = 255 - morph
        count = np.count_nonzero(veins)
        density = count / (ht * wd)
        #print("gray matter density :", density)
        density2 = np.average(veins)/255
        #print("white matter density:", density2)
        density3 = np.amax(veins)/255
        print("density:", (density+density2+density3)/3)

        return ((density+density2+density3)/3)

def describe(image,radius,numPoints):
        eps=1e-7
        lbp = feature.local_binary_pattern(image,numPoints,radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

feature1=[]
feature1.append(ReHo(fmri))
feature1.append(fALFF(fmri))

feature2=[]
feature2.append(seed(fmri))

feature2.append(density(sxc1))
feature2.append(density(sxc2))
feature2.append(density(sxc3))





fea=(np.array(feature2+feature1))
