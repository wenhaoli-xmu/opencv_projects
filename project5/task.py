from cv2 import cv2
import numpy as np
import os

files = os.listdir()

if "paper_2.jpg" not in files:
    path = os.getcwd()
    path += "\\project5\\待检测图像"
    os.chdir(path)

show_process = True

def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

org1 = cv2.imread('paper_2.jpg')

paper1 = cv2.cvtColor(org1, cv2.COLOR_BGR2GRAY)

#Canny边缘检测
paper1 = cv2.Canny(paper1, 80, 150)
if show_process:
    imshow(paper1)

#膨胀操作
kernel = np.ones((3, 3), np.uint8)
paper1 = cv2.dilate(paper1, kernel, iterations=1)
if show_process:
    imshow(paper1)

#轮廓检测
cnts = cv2.findContours(paper1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

#提取面积最大的轮廓
cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

#绘制面积最大的轮廓
show = org1.copy()
show = cv2.drawContours(show, cnts, 0, (0, 255, 0), 1)

if show_process:
    imshow(show)

cnt = []

#步长设置为周长的0.0001倍，一般来说取epsilon = 0.001倍周长
step = 0.0001 * cv2.arcLength(cnts[0], True)
epsilon = step

#不断递增epsilon直到近似所得轮廓正好包含四个点
while len(cnt) != 4:
    cnt = cv2.approxPolyDP(cnts[0], epsilon, True)

    #步增epsilon
    epsilon += step

#绘制轮廓近似的结果
show = org1.copy()
show = cv2.drawContours(show, [cnt], 0, (0, 255, 0), 1)
if show_process:
    imshow(show)

points = np.zeros((4, 2), np.float32)

#将结果转换为np.array
i = 0
for p in cnt:
    (x, y) = p[0]
    points[i][0] = x
    points[i][1] = y
    i += 1

#将四个轮廓点排序
pts = np.zeros((4, 2), np.float32)

res = np.sum(points, axis=1)
pts[0] = points[np.argmin(res)]
pts[2] = points[np.argmax(res)]

res = np.diff(points, axis=1)
pts[1] = points[np.argmin(res)]
pts[3] = points[np.argmax(res)]

#计算边长
w1 = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
w2 = np.sqrt((pts[2][0] - pts[3][0]) ** 2 + (pts[2][1] - pts[3][1]) ** 2)
w = int(max(w1, w2))

h1 = np.sqrt((pts[1][0] - pts[2][0]) ** 2 + (pts[1][1] - pts[2][1]) ** 2)
h2 = np.sqrt((pts[0][0] - pts[3][0]) ** 2 + (pts[0][1] - pts[3][1]) ** 2)
h = int(max(h1, h2))

#目标四个点
dst = np.array([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
], np.float32)

#透视变换
mat = cv2.getPerspectiveTransform(pts, dst)
paper1 = org1.copy()
paper1 = cv2.warpPerspective(paper1, mat, (w, h))
if show_process:
    imshow(paper1)

#切掉四周
(h, w) = paper1.shape[:2]

cuth = int(h * 0.02)
cutw = int(w * 0.02)

paper1 = paper1[:][cuth:h-cuth]
if show_process:
    imshow(paper1)

#预处理
org1 = paper1
paper1 = cv2.cvtColor(paper1, cv2.COLOR_BGR2GRAY)

#直方图均值化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
paper1 = clahe.apply(paper1)
if show_process:
    imshow(paper1)

#自适应二值化
paper1 = cv2.threshold(paper1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
if show_process:
    imshow(paper1)

#闭运算补洞
kernel = np.ones((5, 5), np.uint8)
paper1 = cv2.morphologyEx(paper1, cv2.MORPH_CLOSE, kernel)
if show_process:
    imshow(paper1)

#外轮廓检测
cnts = cv2.findContours(paper1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
show = cv2.drawContours(org1.copy(), cnts, -1, (0, 255, 0), 1)
if show_process:
    imshow(show)

#用于保存保留下来的轮廓
cntsex = []

#上下边界阈值
thresh_lower = 0.8
thresh_upper = 1.2

eps = 1e-6

show = org1.copy()

for cnt in cnts:
    
    cntcopy = cnt.copy()

    #按照h方向坐标对轮廓的所有点排序，找到最大的y
    cntcopy = sorted(cntcopy, key=lambda x: x[0][1], reverse=True)
    maxy = cntcopy[0][0][1]

    #按照w方向坐标对轮廓的所有点排序，找到最大的x
    cntcopy = sorted(cntcopy, key=lambda x: x[0][0], reverse=True)
    maxx = cntcopy[0][0][0]

    #获得椭圆的中心
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    #获得椭圆的长轴和短轴
    a = maxx - x
    b = maxy - y

    if b == 0:
        continue

    ratio = a / b;
    if ratio > 2 or ratio < 0.5:
        continue

    if radius == 0:
        continue
    
    #面积过滤
    areaex = np.pi * a * b
    area   = cv2.contourArea(cnt)

    ratio = area / areaex

    if ratio < thresh_upper and ratio > thresh_lower:
        cntsex.append(cnt)

    show = cv2.drawContours(show, [cnt], 0, (0, 255, 0), 1)
    show = cv2.ellipse(show, center, (int(a), int(b)), 0, 0, 360, (0, 0, 255), 1)

#第二次过滤
cnts = []
maxarea = -1e6

for cnt in cntsex:
    area = cv2.contourArea(cnt)

    if area > maxarea:
        maxarea = area

maxgap = 0.5 * maxarea

cntsex = sorted(cntsex, key=lambda x: cv2.contourArea(x), reverse=True)

prvarea = cv2.contourArea(cntsex[0])
cnts.append(cntsex[0])

for i in range(1, len(cntsex)):
    if abs(prvarea - cv2.contourArea(cntsex[i])) > maxgap:
        break
    cnts.append(cntsex[i])

show = cv2.drawContours(org1.copy(), cnts, -1, (0, 255, 0), 1)
if show_process:
    imshow(show)

#对多个轮廓按照从上到下的顺序排序
cnts = sorted(cnts, key=lambda x: x[0][0][1])

rows = int(len(cnts) / 5)

TAB = ['A', 'B', 'C', 'D', 'E']
ANS = []

#检查每一行（即每一题）的答案
for i in range(rows):
    subcnts = cnts[i*5:(i+1)*5]
    subcnts = sorted(subcnts, key=lambda x: x[0][0][0])

    total = []

    for (j, cnt) in enumerate(subcnts):
        mask = np.zeros(paper1.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1) #-1表示填充

        mask = cv2.bitwise_and(paper1, paper1, mask=mask)
        total.append(cv2.countNonZero(mask))

    idx = np.argmax(np.array(total))
    ANS.append(TAB[idx])

print(ANS)