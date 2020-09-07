from cv2 import cv2
import numpy as np
import os

def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imdiv(img, div):
    (x, y) = img.shape[:2]
    x = int(float(x)/div)
    y = int(float(y)/div)
    img = cv2.resize(img, (y, x))
    return img

files = os.listdir()
print(files)
if 'left.jpg' not in files:
    path = os.getcwd()
    path += '\\project3'
    os.chdir(path)

img1 = cv2.imread('left.jpg')
img2 = cv2.imread('right.jpg')

img1 = imdiv(img1, 5)
img2 = imdiv(img2, 5)

#去掉图像下方的水印
h1 = img1.shape[0]
h2 = img2.shape[0]
h1 = int(h1*0.5)
h2 = int(h2*0.9)
img1 = img1[:][:h1]
img2 = img2[:][:h2]

#将待拼接的图像显示
show = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), np.uint8)
show[0:img1.shape[0], 0:img1.shape[1]] = img1
show[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
imshow(imdiv(show, 1.5))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#特征提取
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

########
# test #
########
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
good = sorted(good, key=lambda x: x.distance)

show = cv2.drawMatches(img1, kp1, img2, kp2, good[:20], None, flags=2)
imshow(imdiv(show, 1.5))
########
# test #
########

#记录特征点的x，y坐标
pos1 = np.float32([kp.pt for kp in kp1])
pos2 = np.float32([kp.pt for kp in kp2])

#特征值匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []

for m, n in matches:

    #通过knn算法筛选优良匹配
    if m.distance < 0.75 * n.distance:

        #记录优良匹配的两个匹配点的index
        good.append((m.trainIdx, m.queryIdx))

#记录匹配点对中左图点的x, y坐标放在pts1中
pts1 = np.float32([pos1[i] for (_, i) in good])

#记录匹配点对中右图点的x, y坐标放在pts2中
pts2 = np.float32([pos2[i] for (i, _) in good])

#通过这些点获得单应性矩阵H（线性变换矩阵）
(H, status) = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)

#变换
res = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
res[0:img1.shape[0], 0:img1.shape[1]] = img1
imshow(res)