import json
import os
from preprocess import preprocess
from utils import getFilePaths, loadData, saveAsPNG, deleteDir

with open('../filepath.json', 'r') as f:
    data = f.read()
    paths = json.loads(data)

filepaths = getFilePaths(paths['data'], flag=1) # all .nii files inside PaloAlto

smri = filepaths[0]
preproc = preprocess(smri, 1, paths['outputs'])
preproc.run()
