import cv2
import os
import time
import numpy as np
from PIL import Image
import threading

path = 'e:\\BAPL-3d-cut\\'
img_list = []
list_dir = os.listdir(path)
# print(int(list_dir[1].split('_')[1].split('.')[0]))
list_dir.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
# list_dir = os.listdir('.\\')
# print(len(list_dir))

# im=Image.open(path+list_dir[])

error_list = []
def convert(start):
    if start+200 > len(list_dir):
        new_list = list_dir[start:len(list_dir)]
    else:
        new_list = list_dir[start:start+200]
    group = start/200
    for index, item in enumerate(new_list):
        new_name = str(item.split('.')[0])
        start_time = time.time()

        print('【'+str(group)+'】'+'('+str(index+1)+'/'+str(len(new_list))+')'+'converted：' + str(item) + ' to ' + new_name +'.png')
        im=Image.open(path+item)
        try:
            im.save(r'E:\BAPL-3d-cut-origin\\'+ new_name +'.png')
        except:
            error_list.append(item)
            pass
        end_time=time.time()
        print('complete！ spend '+str(end_time-start_time) + 's')
        continue

threads = []
t1 = threading.Thread(target=convert,args=(0,))
threads.append(t1)
t2 = threading.Thread(target=convert,args=(200,))
threads.append(t2)
t3 = threading.Thread(target=convert,args=(400,))
threads.append(t3)
t4 = threading.Thread(target=convert,args=(600,))
threads.append(t4)
t5 = threading.Thread(target=convert,args=(800,))
threads.append(t5)
t6 = threading.Thread(target=convert,args=(1000,))
threads.append(t6)
t7 = threading.Thread(target=convert,args=(1200,))
threads.append(t7)
t8 = threading.Thread(target=convert,args=(1400,))
threads.append(t8)



if __name__ == '__main__':
    for t in threads:
        t.setDaemon(True)
        t.start()

    for b in threads:
        b.join()
    
    
    print("all over")
    print(error_list)


