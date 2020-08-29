import os

import numpy as np
import pytesseract
from cv2 import cv2
from PIL import Image


def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imdiv(img, div):
    (x, y) = img.shape[:2]
    x = int(float(x)/div)
    y = int(float(y)/div)
    img = cv2.resize(img, (y, x))
    return img

files = os.listdir()
if 'note.jpg' not in files:
    path = os.getcwd()
    path += '\project2'
    os.chdir(path)

#读取图像
img_rgb = cv2.imread('note.jpg')
img_gray = cv2.imread('note.jpg', 0)
img = img_gray.copy()

#Canny算法检测边缘
img = cv2.Canny(img, 80, 150)
imshow(img)

#提取轮廓
cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts.sort(key=cv2.contourArea, reverse=True)
cnt = cnts[0]
tmp = img_rgb.copy()
show = cv2.drawContours(tmp, cnts, 0, (0, 0, 255), 3)
imshow(show)

#将轮廓转化为矩形，自适应
peri = cv2.arcLength(cnt, True)
step = 0.001
param = 0
approx = cnt

while len(approx) != 4:
    param += step
    approx = cv2.approxPolyDP(cnt, param * peri, True)

points = np.zeros((4, 2), np.float32)
for (i, j) in enumerate(approx):
    points[i] = approx[i][0]

tmp = img_rgb.copy()
show = cv2.drawContours(tmp, [approx], -1, (0, 255, 0), 3)
imshow(show)

#先将四个点排序，找到左上，右上，左下，右下
pts = np.zeros((4, 2), np.float32)
res = np.sum(points, axis=1)
pts[0] = points[np.argmin(res)]
pts[2] = points[np.argmax(res)]
res = np.diff(points, axis=1)
pts[1] = points[np.argmin(res)]
pts[3] = points[np.argmax(res)]

pts = np.array(pts, np.float32)

#计算边长
w1 = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
w2 = np.sqrt((pts[2][0] - pts[3][0]) ** 2 + (pts[2][1] - pts[3][1]) ** 2)
w = int(max(w1, w2))

h1 = np.sqrt((pts[1][0] - pts[2][0]) ** 2 + (pts[1][1] - pts[2][1]) ** 2)
h2 = np.sqrt((pts[0][0] - pts[3][0]) ** 2 + (pts[0][1] - pts[3][1]) ** 2)
h = int(max(h1, h2))

#计算目标图像的尺寸
dst = np.array([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
], np.float32)

#透视变换
mat = cv2.getPerspectiveTransform(pts, dst)
img = img_gray.copy()
img = cv2.warpPerspective(img, mat, (w, h))
imshow(img)

#二值化
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
imshow(img)

#将待检测的二值图像保存成文件
filename = 'buffpic.jpg'
cv2.imwrite(filename, img)

#ocr识别
buf = Image.open(filename)
text = pytesseract.image_to_string(buf)
print(text)
os.remove(filename)