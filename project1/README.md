# 项目：识别银行卡上的数字

---

# 项目心得
## **1. 开运算和闭运算很重要**
一般在二值化处理后，图像可能有毛刺，可以用开运算去掉毛刺。有时候图像有孔洞，可以用闭运算去掉孔洞。需要将一些形状连为一体，也可以采用闭运算，此时卷积盒要稍微大一些。

## **2. 轮廓体取是非常常用的方法**
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

