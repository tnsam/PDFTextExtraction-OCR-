from os import listdir
from os.path import isfile, join
mypath = "D:/fyp/project2/PDFImageExtraction/samplesdata"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlyfiles)

from pdf2image import convert_from_path 

for file in onlyfiles:
    if file.endswith(".pdf"):
        # Store Pdf with convert_from_path function
        images = convert_from_path(pdf_path= 'D:/fyp/project2/PDFImageExtraction/samplesdata/' + file , poppler_path=r'D:\fyp\project2\PDFDataExtraction\poppler-0.68.0_x86\poppler-0.68.0\bin')

        for i in range(len(images)):  
            # print(file)
            # print(file.split('.')[0]) 
            # Save pages as images in the pdf
            filename = 'D:/fyp/project2/PDFImageExtraction/Sample_data/' + file.split('.')[0] + '.' + str(i) +'.png'
            images[i].save(filename, 'JPEG')