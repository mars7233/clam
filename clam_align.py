import os
import cv2
import time
import numpy as np
from PIL import Image
from datetime import datetime

MIN_MATCH_COUNT = 50
# path = './BAPL-3d-cut-origin'
path = 'e:\\BAPL-3d-cut\\'

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = "./output/"+str(now)


img_list = []
list_dir = os.listdir(path)
# print(list_dir)
# list_dir.sort(key= lambda x: int(x.split('.')[0]))
list_dir.sort(key=lambda x: int(x.split('_')[1][:-4]))

# print(list_dir)


def img_match(img1, img2, file1,file2):
    # img1 = cv2.imread(
    #     '/Users/mamingjun/Documents/School/Science/Clam/BAPL-3d-cut-png/1.png')
    # img2 = cv2.imread(
    #     '/Users/mamingjun/Documents/School/Science/Clam/BAPL-3d-cut-png/2.png')
    # gray = img1
    # gray2 = img2

    match_path = output_path+'/match'
    match_folder = os.path.exists(match_path)
    if not match_folder:
        os.makedirs(match_path)
        print("---  new folder "+match_path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    start_time = time.time()
    ret, img1_binary = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY_INV)
    ret, img2_binary = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY_INV)
    # img1_binary = cv2.erode(img1_binary,(7,7))
    # img2_binary = cv2.erode(img2_binary,(7,7))

    # ret, img1_binary = cv2.threshold(img1_binary, 150, 255, cv2.THRESH_BINARY_INV)
    # ret, img2_binary = cv2.threshold(img2_binary, 150, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((2,2),np.uint8)
    img1_open = cv2.morphologyEx(img1_binary,cv2.MORPH_OPEN,k)
    img2_open = cv2.morphologyEx(img2_binary,cv2.MORPH_OPEN,k)

    flag = True
    try:
        # sift = cv2.xfeatures2d.SIFT_create()
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(img2_open, None)
        kp2, des2 = detector.detectAndCompute(img1_open, None)
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
        # matching = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_2[:20], None, flags=2)
        matching = cv2.drawMatchesKnn(img1_open, kp1, img2_open, kp2, good_2[:20], None, flags=2)

        cv2.imwrite(match_path+'/'+ file1+'.png'+ '和'+file2+'.png', matching)

        # if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # H, mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC)  # , cv2.RANSAC, 10)
        H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        print(H)

        # result = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
        result = cv2.warpAffine(img2, H, (img2.shape[1], img2.shape[0]))
        # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # wrap = cv2.warpPerspective(img2, H, (img2.shape[1]+img2.shape[1], img2.shape[0]+img2.shape[0]))
        # result[0:img2.shape[0], 0:img2.shape[1]] = img1

        # rows, cols = np.where(result[:, :, 0] != 0)
        # min_row, max_row = min(rows), max(rows) + 1
        # min_col, max_col = min(cols), max(cols) + 1
        # result = result[min_row:max_row, min_col:max_col, :]  # 去除黑色无用部分

        # alpha_channel = result[:, :, 3]
        # _, mask = cv.threshold(alpha_channel, 254, 255,
        #                        cv.THRESH_BINARY)  # binarize mask
        # color = result[:, :, :3]
        # new_img = cv.bitwise_not(cv.bitwise_not(color, mask=mask))
    except EOFError as e:
        print("Error:"+str(e))
        flag = False


    # 输出
    if np.any(H == 0):
        print("Error")
        flag = False
    else:
        cv2.imwrite(output_path+'/'+file2+'.png', result)
    end_time = time.time()
    print("耗时："+str(end_time-start_time)+"秒")
    return result, flag


def sift_main():
    result_folder = os.path.exists(output_path)
    if not result_folder:
        os.makedirs(output_path)
        print("---  new folder "+output_path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    img1 = cv2.imread('./BAPL-3d-cut-png/1.png', cv2.IMREAD_UNCHANGED)
    img_list = []
    for i, item in enumerate(list_dir):
        if i < len(list_dir):
            item1 = item
            item2 = list_dir[i+1]  # str(int(item.split('.')[0])+1)+'.png'
            print("("+str(i+1)+"/"+str(len(list_dir))+")正在处理第"+item1+"和"+item2)
            if i == 0:
                img1 = img1
            # else:
            #     img1 = cv2.imread('./BAPL-3d-cut-png/'+item1)
            img2 = cv2.imread('./BAPL-3d-cut-png/'+item2)

            # result = img_match(img1, img2, i+1)
            result = img_match(img1, img2, item1.split('.')[0],item2.split('.')[0])
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



sift_main()


def outline_match(img1, img2, index):
    print(2333)

def main():
    result_folder = os.path.exists(output_path)
    if not result_folder:
        os.makedirs(output_path)
        print("---  new folder "+output_path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    img1 = cv2.imread('./BAPL-3d-cut-png/161.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('./BAPL-3d-cut-png/162.png', cv2.IMREAD_UNCHANGED)

    ret, img1_binary = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY_INV)
    ret, img2_binary = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY_INV)

    # cv2.imwrite(output_path+'/img1_binary.png', img1_binary)
    # cv2.imwrite(output_path+'/img2_binary.png', img2_binary)
    k = np.ones((3,3),np.uint8)
    img1 = cv2.morphologyEx(img1_binary,cv2.MORPH_OPEN,k)
    img2 = cv2.morphologyEx(img2_binary,cv2.MORPH_OPEN,k)
    cv2.imwrite(output_path+'/img_open1.png', img1)
    cv2.imwrite(output_path+'/img_open2.png', img2)

    img = img_match(img1,img2,1,2)[0]
    

    # gray = cv2.cvtColor(img1, cv2.COLOR_BAYER_BG2GRAY)
    # img = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    # # img.fill(255)
    # # cv2.imwrite(output_path+'/img.png', img)
    # ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # # print(ret)

    # cv2.imwrite(output_path+'/binary.png', binary)
    # binary = delete_min2(binary)
    # # blur = cv2.GaussianBlur(binary, (5, 5), 0, 0)
    # cv2.imwrite(output_path+'/erosion.png', binary)
    # image,contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # rect = cv2.minAreaRect(contours[0])
    # # print(rect)
    # points = cv2.boxPoints(rect)
    # points = np.int0(points)
    # image= cv2.drawContours(img,[points],-1,(255,255,255),3)
    # cv2.imwrite(output_path+'/1.png', image)

# main()
