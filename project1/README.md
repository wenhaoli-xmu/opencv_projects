# 项目：识别银行卡上的数字

---

# 项目心得
## **开运算和闭运算**
一般在二值化处理后，图像可能有毛刺，可以用开运算去掉毛刺。有时候图像有孔洞，可以用闭运算去掉孔洞。需要将一些形状连为一体，也可以采用闭运算，此时卷积盒要稍微大一些。

## **轮廓体取**
在轮廓提取中，想要遍历轮廓，可以采用以下方法：

```python
rects = []

for (i, c) in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    rects.append((x, y, w, h))
```

枚举 `enumerate` 是一个很好用的方法，它还可以用于遍历外接矩阵 `rects`
```python
for (i, (x, y, w, h)) in enumerate(rects):
    #进行相应处理，例如绘图，例如提取子图
```

## **Sobel算子求梯度**
在普通的sobel算子计算梯度的过程中，常常是计算x方向和y方向两个方向，但在本例中只求了一个方向，将一个方向上的梯度转化为灰度图像，可以采用以下方法:
```python
#sobel算子求梯度
img = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

#绝对值+转化成图像
img = np.absolute(img)
(minVal, maxVal) = (np.min(img), np.max(img))
img = (255 * ((img - minVal) / (maxVal - minVal)))
img = img.astype('uint8')
```
## **礼帽操作 Tophat**
礼帽操作原本是获取图像中的噪声部分，但是由于在银行卡上，一些细小的数字，类似于噪声，所以采用礼帽操作可以提取这些细小的数字。
