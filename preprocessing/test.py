import json
import os
from preprocess import preprocess
from utils import getFilePaths, loadData, saveAsPNG, deleteDir

with open('../filepath.json', 'r') as f:
    data = f.read()
    paths = json.loads(data)

filepaths = getFilePaths(paths['data'], 'PaloAlto', functional=False) # all .nii files inside directory
for f in filepaths:
    if f.endswith("_anonymized.nii.gz"):

        fsplit = f.split("/")
        subj = fsplit[-3]
        opdir = os.path.join(paths['outputs'], subj) 
        if os.path.exists(opdir):
            os.remove(opdir)
        os.mkdir(opdir)

        # img = loadData(f, verbose=True) # loading first subject's sMRI

        # preproc = preprocess(f, imtype='structural', savePath=os.path.join(paths['outputs'], subj))
        # preproc.run()
