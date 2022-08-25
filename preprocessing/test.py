import json
import os
from preprocessing.utils import getFilePaths, skullStrip, tiltCorrection, loadData, cropImage, showScan, addPadding, biasFieldCorrection, intensityNormalisation
import SimpleITK as sitk

with open('../filepath.json', 'r') as f:
    data = f.read()
    paths = json.loads(data)

filepaths = getFilePaths(paths['data'], 'PaloAlto', functional=False)

img = loadData(filepaths[0], verbose=True)

# intensityNormalisation(filepaths[0])

load = loadData('/home/arunav/Assets/outputs/in_out.nii')
showScan(load)
