import os
import sys
import pydicom
from tqdm import tqdm
import numpy as np
import cv2
import SimpleITK

reader = SimpleITK.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
print(dicom_names)
reader.SetFileNames(dicom_names)
image = reader.Execute()

img_array = SimpleITK.GetArrayFromImage(image)
print(img_array.shape)
print(img_array.max(), img_array.min(), img_array.dtype)

tot = img_array.shape[0]
img_array[img_array < 0] = 0

img_array = img_array / 6000.
img_array = img_array * 255
img_array = img_array.astype(np.uint8)

path = sys.argv[1].replace('dicom', 'image')
os.system("mkdir -p %s"%path)
for i in range(tot):
    fname = '%s/%d.png'%(path, i)
    cv2.imwrite(fname, img_array[i])
    print(i, tot)

