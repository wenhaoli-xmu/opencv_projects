from cv2 import cv2
import numpy as np
import os

def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sortContours(rect):
    for i in range(len(rect) - 1):
        for j in range(i, len(rect)):
            if rect[i][0] > rect[j][0]:
                tmp = np.copy(rect[i])
                rect[i] = rect[j]
                rect[j] = tmp
    return rect

files = os.listdir()
if 'template' not in files:
    path = os.getcwd()
    path += '\project1'
    os.chdir(path)

tem_rgb = cv2.imread('template.png')
img_rgb = cv2.imread('credit_card.png')

tem_gray = cv2.imread('template.png', 0)
img_gray = cv2.imread('credit_card.png', 0)

tem = np.copy(tem_gray)
img = np.copy(img_gray)

#对模板预处理
ret, tem = cv2.threshold(tem, 127, 255, cv2.THRESH_BINARY_INV)

#获取边界
binary, contours, hierarchy = cv2.findContours(tem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#获取外接长方形
copy = tem_rgb.copy()
rects = []
for i in range(len(contours)):
    (x, y, w, h) = np.array(cv2.boundingRect(contours[i]))
    copy = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    rects.append((x, y, w, h))

#将外接长方形排序
rects = sortContours(rects)

#获取每个数字模板，保存在temps中
temps = {}
for i in range(10):
    (x, y, w, h) = tuple(rects[i])
    roi = tem[y:y+h, x:x+w]
    roi = cv2.resize(roi, (60, 100))
    temps[i] = roi

#对输入图像处理

#礼帽操作
kernel = np.ones((9, 9), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

#sobel算子求梯度
img = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

#绝对值+转化成图像
img = np.absolute(img)
(minVal, maxVal) = (np.min(img), np.max(img))
img = (255 * ((img - minVal) / (maxVal - minVal)))
img = img.astype('uint8')

#对图像进行闭运算
kernel = np.ones((8, 8), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#自适应二值化
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#再对图像进行闭运算
kernel = np.ones((12, 12), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#对图像进行开运算，去掉毛刺
kernel = np.ones((4, 4), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#轮廓检测
contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
tmp = img_rgb.copy()
show = cv2.drawContours(tmp, contours, -1, (0, 0, 255), 2)
rects = []
cnts = []

#遍历轮廓
for (i, c) in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4 and w < 80 and h < 20:
        rects.append((x, y, w, h))
        cnts.append(c)

tmp = img_rgb.copy()
show = cv2.drawContours(tmp, cnts, -1, (0, 0, 255), 1)

#给轮廓排序
rects = sortContours(rects)

#遍历轮廓
nums = []
for (i, (gx, gy, gw, gh)) in enumerate(rects):
    buf = img_gray[gy-5:gy+gh+5, gx-5:gx+gw+5]
    buf = cv2.threshold(buf, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #闭运算
    # kernel = np.ones((2, 2), np.uint8)
    # buf = cv2.morphologyEx(buf, cv2.MORPH_CLOSE, kernel)
    # imshow(buf)

    #轮廓提取
    cnts = cv2.findContours(buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    buff = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        buff.append((x, y, w, h))
    buff = sortContours(buff)
    for (i, (x, y, w, h)) in enumerate(buff):
        nums.append(cv2.resize(buf[y:y+h, x:x+w], (60, 100)))

ans = []


for (i, img) in enumerate(nums):
    scores = []

    for (j, tem) in temps.items():
        res = cv2.matchTemplate(img, tem, cv2.TM_CCOEFF_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(res)
        scores.append(score)

    ans.append(str(np.argmax(scores)))

ans = ''.join(ans)
print(ans)