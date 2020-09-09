# 项目：停车场车位识别

---

# **解决过程如下**
## **形态学操作**
对原始图像预处理操作，然后第一步给某一帧图像套上mask遮罩，原始图像和遮罩如下图所示：
![img1](https://github.com/RainFromCN/opencv_projects/blob/master/project4/frame.jpg)
![img2](https://github.com/RainFromCN/opencv_projects/blob/master/project4/mask.jpg)

然后对图片二值化等基本操作，得到的结果进行礼帽操作（TOPHAT），目的是去除干扰的白色车辆，效果如下图所示：
![img3](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic1.png)

## **垂直直线检测**
对图像进行霍夫变换，选择较大的阈值，将图像中的垂直直线检测出来，并使用`abs(y2 - y1) >= 20 and abs(x2 - x1) <= 10`的方法进行粗略的结果筛选，具体代码如下：
```python
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
```
处理结果如下：
![img4](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic2.png)

之后对图像进行闭运算 + 开运算，去除未填补满的小洞以及毛刺部分，然后使用Canny边缘检测，检测出图像中的轮廓部分，处理结果如下：
![img5](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic3.png)

之后对边界进行轮廓检测，并找到外接矩形，然后即可获得每一大列（总共10列）的中心位置

## **水平直线检测**
依然使用霍夫变换进行水平直线检测，此时选择较小的阈值，具体的代码如下：
```python
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
```
处理结果如下：
![img6](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic4.png)
之后进行排序和聚类，使得其顺序为从上到下，并且分为十大簇，每一簇代表一大列

## **记忆化搜索择优**
观察上一步得到的结果，容易看出绿色的线有重合，而且非常的杂乱还有空缺，此时需要使用记忆化搜索进行择优，具体的代码比较复杂，如下：
```python
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
```
处理的结果如下：
![img7](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic5.png)

## **插值和对齐**
记忆化搜索出来的结果没有重叠的部分，而且都是匹配比较优良的组合，但是其没有左右对齐，比较杂乱，而且中间有空隙，
需要使用插值算法填补其中的空隙，使用对齐算法将每一簇的横线对齐

具体的代码如下：
```python
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
```
处理结果如下：
![img8](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic7.png)

最后进行图像的切分，切分结果如下：
![img9](https://github.com/RainFromCN/opencv_projects/blob/master/project4/pic8.png)

















