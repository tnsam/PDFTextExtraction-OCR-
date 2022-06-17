import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

for imageName in glob.glob('D:/fyp/project2/PDFImageExtraction/bca/*.png'): #assuming JPG
    print(imageName)
    print(imageName.rsplit('/', 1)[-1])
    img = cv2.imread(imageName)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite('D:/fyp/project2/PDFImageExtraction/preprocessed_bca/' + imageName.rsplit('\\', 1)[-1], thresh)
    # plt.imshow(thresh)
    # plt.show()
    print("\n")