import os
import cv2
import time
import numpy as np
from PIL import Image
from datetime import datetime


program_start = time.time()

MIN_MATCH_COUNT = 50
# path = './BAPL-3d-cut-origin'
path = 'e:\\BAPL-3d-cut-origin\\'

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# output_path = "./output/"+str(now)
output_path = 'E:\\BAPL-3d-output\\' + str(now)
output_path = 'E:\\BAPL-3d-output\\2019-12-24-17-22-40'


img_list = []
list_dir = os.listdir(path)

list_dir.sort(key=lambda x: int(x.split('_')[1][:-4]))
list_dir = list_dir[157:]

# print(list_dir)


def img_match(img1, img2, file1, file2,pre=True):
    start_time = time.time()
    
    match_path = output_path+'\\match'
    match_folder = os.path.exists(match_path)
    if not match_folder:
        os.makedirs(match_path)
        print("---  new folder "+match_path+"...  ---")
    origin_path = output_path+'\\origin'
    origin_folder = os.path.exists(origin_path)
    if not origin_folder:
        os.makedirs(origin_path)
        print("---  new folder "+origin_path+"...  ---")

    ret, img1_binary = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY_INV)
    ret, img2_binary = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY_INV)
    # img1_binary = cv2.erode(img1_binary,(7,7))
    # img2_binary = cv2.erode(img2_binary,(7,7))
    k1 = np.ones((2, 2), np.uint8)
    k2 = np.ones((2, 2), np.uint8) 
    if img1.shape[1] < 5000 and img2.shape[1] > 5000:
        k2 = np.ones((1, 1), np.uint8)
        k2 = np.ones((5, 5), np.uint8)
        print("k=5*5")

    img1_open = cv2.morphologyEx(img1_binary, cv2.MORPH_OPEN, k1)
    img2_open = cv2.morphologyEx(img2_binary, cv2.MORPH_OPEN, k2)

    flag = True

    # sift = cv2.xfeatures2d.SIFT_create()
    # detector = cv2.ORB_create()
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(img2_open, None)
    kp2, des2 = detector.detectAndCompute(img1_open, None)
    if img1.shape[1] < 5000 and img2.shape[1] > 5000:
        kp1, des1 = detector.detectAndCompute(img2, None)
        kp2, des2 = detector.detectAndCompute(img1, None)
        
    # img1=cv2.drawKeypoints(gray,kp1,img)
    # img2=cv2.drawKeypoints(gray2,kp2,img2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # matches = bf.match(des1, des2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # cv2.drawMatchesKnn expects list of lists as matches
    good_2 = np.expand_dims(good, 1)
    matching = cv2.drawMatchesKnn(img1_open, kp1, img2_open, kp2, good_2[:20], None, flags=2)

    # cv2.imwrite(match_path+'/'+ file1+'.png'+ '和'+file2+'.png', matching)
    cv2.imwrite(match_path+'\\' + file1+'.png' +' and '+file2+'.png', matching)

    # if len(good) > MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    print(H)

    result_origin = cv2.warpAffine(img2, H, (img2.shape[1], img2.shape[0]))
    result = cv2.warpAffine(img2_open, H, (img2_open.shape[1], img2_open.shape[0]))

    if np.any(H == 0):
        print("Error")
        flag = False
        return result_origin, flag
    else:    
        # 输出    
        # cv2.imwrite(output_path+'/'+file2+'.png', result)
        result1 = test_add(result)
        cv2.imwrite(output_path+'\\'+file2+'.png', result1)
        cv2.imwrite(origin_path+'\\'+file2+'.png', result_origin)
        end_time = time.time()
        print("该操作耗时："+str(round(end_time-start_time, 2))+"秒, " + "程序已运行了："+str(round((end_time-program_start)/60, 2))+"分")
        return result_origin, flag
    


def sift_main():
    result_folder = os.path.exists(output_path)
    if not result_folder:
        os.makedirs(output_path)
        print("---  new folder "+output_path+"...  ---")

    img1 = cv2.imread('e:\\BAPL-3d-cut-origin\\'+list_dir[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread('E:\\BAPL-3d-output\\2019-12-24-17-22-40\\origin\\' +list_dir[0], cv2.IMREAD_UNCHANGED)

    img_list = []
    for i, item in enumerate(list_dir):
        if i < len(list_dir):
            item1 = item
            item2 = list_dir[i+1]
            print("("+str(i+1)+"/"+str(len(list_dir))+")正在处理第"+item1+"和"+item2)
            if i == 0:
                img1 = img1

            img2 = cv2.imread('e:\\BAPL-3d-cut-origin\\' +item2.split('.')[0]+'.png', cv2.IMREAD_UNCHANGED)
            result = img_match(img1, img2, item1.split('.')
                               [0], item2.split('.')[0])
            if np.any(result[1] == False):
                try:
                    print(len(img_list))
                    img1 = img_list[len(img_list)-2]
                    continue
                except EOFError as e:
                    print("Error:"+str(e))
                    continue
            else:
                img1 = result[0]
                img_list.append(img1)


def test_add(img_n):
    img1 = np.zeros((8000, 8000), np.uint8)
    rows, cols = img_n.shape
    img1[1000:rows+1000, 1000:cols+1000] = img_n

    return img1


def main():
    result_folder = os.path.exists(output_path)
    if not result_folder:
        os.makedirs(output_path)
        print("---  new folder "+output_path+"...  ---")

    img1 = cv2.imread(
        'E:\\BAPL-3d-output\\2019-12-24-17-22-40\\origin\\Stitched Image_789.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(
        'e:\\BAPL-3d-cut-origin\\Stitched Image_805.png', cv2.IMREAD_UNCHANGED)

    ret, img1_binary = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY_INV)
    ret, img2_binary = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY_INV)


    k = np.ones((2,2),np.uint8)
    k2 = np.ones((6,6),np.uint8)
    img1_open = cv2.morphologyEx(img1_binary,cv2.MORPH_OPEN,k)
    img2_open = cv2.morphologyEx(img2_binary,cv2.MORPH_OPEN,k2)
    cv2.imwrite(output_path+'/img_open1.png', img1_open)
    cv2.imwrite(output_path+'/img_open2.png', img2_open)

    img = img_match(img1,img2,'1','2')[0]




sift_main()
# main()
