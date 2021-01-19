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

mask = cv2.imread('mask.jpg', 0)
org = cv2.imread('frame.jpg')

files = os.listdir()
if "parking_video.mp4" not in files:
    path = os.getcwd()
    path += "\\project4"
    os.chdir(path)

imshow(org)

#=================================================================================================#
#预处理
#BGR空间下的阈值化
lower = np.uint8([120, 120, 120])
upper = np.uint8([255, 255, 255])
img = cv2.inRange(org, lower, upper)
imshow(img)

# #中值滤波
# img = cv2.medianBlur(img, 3)
# imshow(img)

#将原图像的背景去掉
img = cv2.bitwise_and(org, org, mask=img)
imshow(img)

#灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(img)

#套上遮罩
img = cv2.bitwise_and(mask, img)
imshow(img)

#礼帽操作
kernel = np.ones((3, 3), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
imshow(img)

#中值滤波
img = cv2.medianBlur(img, 3)
imshow(img)
 
#=================================================================================================#
#垂直直线检测
#霍夫变换
lines = cv2.HoughLinesP(img, rho = 1, theta=np.pi, threshold=5, minLineLength=100, maxLineGap=40)

img1 = np.zeros(org.shape, np.uint8)

cleaned = []
for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y2 - y1) >= 20 and abs(x2 - x1) <= 10:
            cleaned.append((x1, y1, x2, y2))
            cv2.line(img1, (x1, y1), (x2, y2), [255, 255, 255], 2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
imshow(img1)

#闭运算+开运算
kernel = np.ones((11, 11), np.uint8)
img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
imshow(img1)

#canny边缘检测
img1 = cv2.Canny(img1, 80, 150)
imshow(img1)

#轮廓检测
center_line = []
edges = []
show = org.copy()
cnts = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ex = 7
    x -= ex
    w += ex * 2
    center_line.append((int(x + w / 2), int(y + h/2), h))
    edges.append((x, x + w))
    show = cv2.rectangle(show, (x, y), (x + w, y + h), (0, 0, 255), 2)
center_line = sorted(center_line, key=lambda x: x[0])
imshow(show)

#=================================================================================================#
#水平直线检测
#霍夫变换
lines = cv2.HoughLinesP(img, rho = 0.1, theta=np.pi/10, threshold=18, minLineLength=10, maxLineGap=7)

show = org.copy()

cleaned = []
for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(x1 - x2) > 25 and abs(y1 - y2) < 10:
            cleaned.append((x1, y1, x2, y2))
            show = cv2.line(show, (x1, y1), (x2, y2), [0, 255, 0], 1)

for (x, y, h) in center_line:
    (x1, y1, x2, y2) = (x, y - h/2, x, y + h/2)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    show = cv2.line(show, (x1, y1), (x2, y2), (0, 0, 255), 1)
imshow(show)

#排序+聚类
cleaned = sorted(cleaned, key=lambda x: x[0])

break_dot = [0, ] 
idx = []
cnt = 0
thresh = 25

for i in range(1, len(cleaned)):
    x = cleaned[i][0]
    prvx = cleaned[i-1][0]
    if abs(x - prvx) > thresh:
        cnt += 1
        break_dot.append(i)
    idx.append(cnt)

break_dot.append(len(cleaned))

for i in range(len(break_dot) - 1):
    buf = cleaned[break_dot[i]:break_dot[i+1]]
    buf = sorted(buf, key=lambda x: x[1])
    cleaned[break_dot[i]:break_dot[i+1]] = buf
    
#=================================================================================================#
#记忆化搜索择优

#用于储存最终结果
ans = [[] for i in range(len(break_dot) - 1)]

#动态规划数组
dp = []

#缓存变量
buf = []

#用于计算路径path的变量
father = {}
path = []

#寻找近邻函数
err = 2.5
def findnei(idx, y, multi):
    
    y1_upper = y + 15.5*multi + err
    y1_lower = y + 15.5*multi - err
    y2_upper = y - 15.5*multi + err
    y2_lower = y - 15.5*multi - err

    upper = []
    lower = []

    for i in range(len(buf)):
        y = buf[i][1]
        if y > y1_lower and y < y1_upper:
            upper.append(i)
        elif y > y2_lower and y < y2_upper:
            lower.append(i)
    
    return (upper, lower)

#深度优先搜索
def dfs(idx, i):

    #查表
    if dp[i] >= 0:
        return dp[i]

    #取出临近点
    y1 = buf[i][1]
    for j in range(5):
        nei = findnei(idx, y1, j+1)[1]
        if len(nei) != 0:
            break

    #如果没有临近点
    if len(nei) == 0:
        dp[i] = 1
        father[i] = None
        return dp[i]

    #找到最优临近点的编号和DP值
    MAX = -1e6
    MAX_IDX = -1

    #状态转移
    for j in range(len(nei)):
        if dfs(idx, nei[j]) > MAX:
            MAX = dfs(idx, nei[j]) + 1
            MAX_IDX = nei[j]

    #记录father
    dp[i] = MAX
    father[i] = MAX_IDX
    return dp[i]

# memdfs
for i in range(10):
    buf = cleaned[break_dot[i]:break_dot[i+1]]

    #初始化
    dp = [-1 for i in range(len(buf))]
    
    #记忆化搜索
    MAX = -1e6
    for j in range(len(buf)-1, -1, -1):
        if dfs(i, j) > MAX:
            MAX = dfs(i, j)

            #计算path路径
            x = father[j]
            path = []
            path.append(j)
            while x != None:
                path.append(x)
                x = father[x]
    
    #计算ans
    path = sorted(path)
    for j in path:
        ans[i].append(buf[j])
    
#搜索结果可视化
clr = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
show = np.copy(org)
for i in range(len(break_dot) - 1):
    for (x1, y1, x2, y2) in ans[i]:
        show = cv2.line(show, (x1, y1), (x2, y2), clr[i % 3], 2)
imshow(show)

#=================================================================================================#
#插值+对齐优化

#对齐
ex = 30
for i in range(len(ans)):
    (x0, y0, _) = center_line[i]
    for j in range(len(ans[i])):
        (x1, y1, x2, y2) = ans[i][j]
        x1 = x0 - ex
        x2 = x0 + ex
        ans[i][j] = (x1, y1, x2, y2)

#对齐结果显示
show = np.copy(org)
for i in range(len(break_dot) - 1):
    for (x1, y1, x2, y2) in ans[i]:
        show = cv2.line(show, (x1, y1), (x2, y2), (0, 0, 255), 2)
imshow(show)

#插值
err = 2.5
def gap(y1, y2):
    delta = abs(y1 - y2)
    mul = (delta + err) / 15.5
    return int(mul) 

for i in range(len(ans)):

    buf = []

    for j in range(len(ans[i]) - 1):
        (x1, y1, x2, y2) = ans[i][j]
        (_, y, _, _) = ans[i][j+1]
        
        cnt = gap(y, y1)
        for k in range(cnt):
            newy = int(y1 + 15.5 * k)
            buf.append((x1, newy, x2, newy))
    buf.append(ans[i][len(ans[i]) - 1])

    ans[i] = buf

#绘制
show = np.copy(org)
for i in range(len(break_dot) - 1):
    for (x1, y1, x2, y2) in ans[i]:
        show = cv2.line(show, (x1, y1), (x2, y2), (0, 0, 255), 2)
imshow(show)

#=================================================================================================#
#切分

w, h = 30, 15
pic = []

for i in range(len(ans)):
    for j in range(len(ans[i])):
        (x1, y1, _, _) = ans[i][j]
        x2, y2 = x1 + w, y1 + h
        pic.append((x1, y1, x2, y2))
        x1 = x1 + w
        x2 = x2 + w
        pic.append((x1, y1, x2, y2))

prv = os.getcwd()
p = prv + "\\parking_pics"
os.chdir(p)

show = np.copy(org)
cnt = 0
for (x1, y1, x2, y2) in pic:
    show = cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 1)
    subpic = org[y1:y2, x1:x2]
    s = "pos_" + str(cnt) + ".jpg"
    cv2.imwrite(s, subpic)
    cnt += 1
imshow(show)

os.chdir(prv)

#=================================================================================================#