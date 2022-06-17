# # clone YOLOv5 repository
# git clone https://github.com/ultralytics/yolov5  # clone repo
# cd yolov5
# git reset --hard 886f1c03d839575afecb059accf74296fad395b6

# pip install -r requirements.txt

from datetime import time
import torch

from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#download dataset from roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="MmrcvDjjFaFtANipcAx5")
project = rf.workspace().project("cimb2-fuyu0")
dataset = project.version(1).download("yolov5")

print("dataset: ", {dataset.location})

# cat 'D:/fyp/project2/PDFImageExtraction/yolov5/cimb2-2/data.yaml'

import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#cd yolov5
# cat models/yolov5s.yaml

# #train the model
#cd yolov5
# python D:\fyp\project2\PDFImageExtraction\yolov5\train.py --img 416 --batch 16 --epochs 100 --data 'D:/fyp/project2/PDFImageExtraction/yolov5/cimb2-2/data.yaml' --cfg ./models/custom_yolov5s.yaml --weights 'yolov5s.pt' --name yolov5s_results  --cache

# #export trained weights
# cp D:\fyp\project2\PDFImageExtraction\yolov5\runs\train\yolov5s_results14\weights\best.pt cimb_row.pt

# #perform testing

#cimb
# python detect.py --weights 'cimb_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_cimb --save-txt

#bni
# python detect.py --weights 'bni_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_bni --save-txt

#maybank
# python detect.py --weights 'maybank_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_maybank --save-txt

#bca
# python detect.py --weights 'bca_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_bca --save-txt

#mandiri
# python detect.py --weights 'mandiri_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_mandiri --save-txt

#permata
# python detect.py --weights 'permata_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_permata --save-txt

#BRI
# python detect.py --weights 'bri_row.pt' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\crop_bri --save-txt