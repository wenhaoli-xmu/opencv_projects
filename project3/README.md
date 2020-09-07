# 项目：拼接两个图像

---

# 技术关键词
## **SIFT特征提取**
对原始的图像，左图像和右图像分别进行特征提取

原始图像如下图所示
![orgimage](https://github.com/RainFromCN/opencv_projects/blob/master/project3/org.png)

使用以下代码进行特征提取
```python
#特征提取
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```

## **特征匹配**
对于两幅图像的特征进行匹配，使用1对k（k=2）匹配，并使用0.75的阈值筛除不优良的匹配对，具体的代码如下
```python
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
```

下面的这个图片对匹配结果进行了大致的演示（只绘制了前20对特征匹配点）
![process](https://github.com/RainFromCN/opencv_projects/blob/master/project3/process.png)

## **单应性矩阵透视变换**
计算单应性矩阵有两种方法，其中
- `getPerspectiveTranform`用的是SVD分解，只会拿前四个点计算
- `findHomography`则会拿一堆点(>=4)进行计算（不断从一堆点中重复拿出4个点计算结果，再用一些优化算法RANSAC取筛选出最优解）

具体的代码如下：
```python
#通过这些点获得单应性矩阵H（线性变换矩阵）
(H, status) = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)

#变换
res = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
res[0:img1.shape[0], 0:img1.shape[1]] = img1
imshow(res)
```

处理结果如下图所示：
![res](https://github.com/RainFromCN/opencv_projects/blob/master/project3/res.png)
