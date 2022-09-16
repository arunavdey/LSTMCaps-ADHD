import json
import os
from preprocess import preprocess
from utils import getFilePaths, loadData, saveAsPNG, deleteDir
import nibabel as nib

with open('../filepath.json', 'r') as f:
    data = f.read()
    paths = json.loads(data)

filepaths = getFilePaths(paths['data'], flag=2) # all .nii files inside PaloAlto

fmri = filepaths[0]
func = nib.load(fmri)
header = func.header
print(header.get_zooms())
# preproc = preprocess(smri, 1, paths['outputs'])
# preproc.run()
