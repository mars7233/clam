import os
import cv2
import numpy as np


def test_add(img_n):
    img1 = np.zeros((8000, 8000,3), np.uint8)
    rows, cols,s = img_n.shape
    img1[0:rows, 0:cols] = img_n
    return img1


path = 'E:\\BAPL-3d-output\\2019-12-21-18-28-33'
png_list = os.listdir(path)
png_list.remove('match')

print(len(png_list))

for item in png_list:
    img = cv2.imread(path+'\\'+item)
    # print(img)
    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    if(rows != 8000 and cols != 8000):
        img = test_add(img)
        cv2.imwrite(
            'C:\\Users\\mars\\Documents\\work\\clam-align\\output\\'+item,img)
        print(item)
