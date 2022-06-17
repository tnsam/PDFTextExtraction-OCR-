# # clone YOLOv5 repository
# git clone https://github.com/ultralytics/yolov5  # clone repo
# cd yolov5
# git reset --hard 886f1c03d839575afecb059accf74296fad395b6

# # pip install -r requirements.txt

from datetime import time
import torch

from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#download dataset from roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="MmrcvDjjFaFtANipcAx5")
project = rf.workspace().project("pdf-yfbrk")
dataset = project.version(1).download("yolov5")

print("dataset: ", {dataset.location})

# cat 'D:/fyp/project2/PDFImageExtraction/yolov5/PDF-1/data.yaml'

# import yaml
# with open(dataset.location + "/data.yaml", 'r') as stream:
#     num_classes = str(yaml.safe_load(stream)['nc'])

#cd yolov5
# cat models/yolov5s.yaml

# #train the model
#cd yolov5
# python D:\fyp\project2\PDFImageExtraction\yolov5\train.py --img 416 --batch 16 --epochs 100 --data 'D:/fyp/project2/PDFImageExtraction/yolov5/PDF-1/data.yaml' --cfg ./models/custom_yolov5s.yaml --weights 'yolov5s.pt' --name yolov5s_results  --cache

# #export trained weights
# cp D:\fyp\project2\PDFImageExtraction\yolov5\runs\train\yolov5s_results14\weights\best.pt all_invoice

# #perform testing
# python D:\fyp\project2\PDFImageExtraction\yolov5\detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source Sample_output --save-txt
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\Sample_output --save-txt

#bca
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_bca --save-txt

#bni
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_bni --save-txt

#bri
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_bri --save-txt

#cimb
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_cimb --save-txt

#mandiri
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_mandiri --save-txt

#maybank
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_maybank --save-txt

#permata
# python detect.py --weights 'all_invoice' --img 416 --conf 0.4 --source D:\fyp\project2\PDFImageExtraction\preprocessed_permata --save-txt


#save the cropped content in a folder for row identification
import glob
import cv2
import numpy as np

for imageName in glob.glob('D:/fyp/project2/PDFImageExtraction/yolov5/runs/detect/exp7/labels/*_content.png'): #assuming JPG
    print(imageName)
    print(imageName.rsplit('\\', 1)[-1])
    img = cv2.imread(imageName)
    cv2.imwrite('D:/fyp/project2/PDFImageExtraction/crop_permata/' + imageName.rsplit('\\', 1)[-1], img)
    print("\n")

# import glob
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# for imageName in glob.glob('D:/fyp/project2/PDFImageExtraction/yolov5/runs/detect/exp16/labels/*_content.png'): #assuming JPG
#     print(imageName)
#     print(imageName.rsplit('/', 1)[-1])
#     img = cv2.imread(imageName)

#     width = 416
#     height = 416
#     dim = (width, height)
    
#     # resize image
#     resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
#     print('Resized Dimensions : ',resized.shape)

#     gray = cv2.cvtColor(np.array(resized), cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     cv2.imwrite('D:/fyp/project2/PDFImageExtraction/preprocessed_crop_cimb/' + imageName.rsplit('\\', 1)[-1], thresh)
#     # plt.imshow(thresh)
#     # plt.show()
#     print("\n")