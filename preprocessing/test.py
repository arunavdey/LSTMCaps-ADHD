import json
import os
from utils.utils import getFilePaths
from utils.utils import loadData
from utils.utils import skullStrip
from utils.utils import showScan

with open('../filepath.json', 'r') as f:
    data = f.read()
    paths = json.loads(data)

filepaths = getFilePaths(paths['data'], 'PaloAlto', functional = False)

img_load = loadData(filepaths[0])
showScan(img_load)
# for file in filepaths:
#     img_load = loadData(file)
#     showScan(img_load)
    
