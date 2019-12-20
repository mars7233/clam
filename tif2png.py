import cv2
import os
import time
import numpy as np
from PIL import Image

path = '/Volumes/Backup Plus/BAPL-3d-cut/'
img_list = []
list_dir = os.listdir(path)
# print(int(list_dir[1].split('_')[1].split('.')[0]))
list_dir.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

for index, item in enumerate(list_dir):
    start_time = time.time()
    im = Image.open(path+item)
    im.save('./BAPL-3d-cut-png/'+str(index+1)+'.png')
    end_time = time.time()
    print('('+str(index+1)+'/'+str(len(list_dir))+')'+'converted：' +
          str(index+1)+'.png, spend '+str(end_time-start_time) + 's')
