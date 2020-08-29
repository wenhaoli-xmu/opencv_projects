# 项目：OCR识别账单上的文字

---

# 技术关键词
## **Canny边缘检测+轮廓提取**
直接进行轮廓提取，效果可能非常不好，轮廓提取搭配Canny边缘检测使用，可以大大减少提取的轮廓数量，从而更容易筛选出需要的轮廓

在提取轮廓的时候，注意可以通过面积，周长等方法进行筛选，利用`sort`函数对所有轮廓的面积排序的代码如下：
```python
cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts.sort(key=cv2.contourArea, reverse=True)
cnt = cnts[0]
```
则`cnt`就是面积最大的那个轮廓

## **抽取角点**
所需要的轮廓通常是一系列点的集合，如果我们需要把边界处理成一个规规矩矩的矩阵，就需要使用**抽稀算法**，把类似矩阵的轮廓的四个角点抽离出来。在具体使用过程中，考虑到鲁棒性，可以采用**递增的方法**，不断增大阈值，直到刚好抽离出四个角点，具体代码如下：
```python
peri = cv2.arcLength(cnt, True) #轮廓周长
step = 0.001 #步长是0.001
param = 0 #阈值初始值是0
approx = cnt

while len(approx) != 4:
    param += step
    approx = cv2.approxPolyDP(cnt, param * peri, True)
```
## **透视变换**
在已经获得规规矩矩的轮廓四边形的四个角点后，需要使用透视变换进行变换，获取两个形状的透视变换矩阵，需要使用函数`cv2.getPerspectiveTransform(pts, dst)`其中`pts`是原图像轮廓四边形的四个角点，`dst`用于描述目标图像的尺寸（也是四个角点）

### 具体操作如下：
- 首先获取原轮廓四边形的**最大宽**和**最大高**，
- 然后获取变换矩阵M
- 最后使用变换矩阵M得到变换后的图像

### 具体代码如下：
```python
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
```
## **OCR识别**
ocr识别需要使用两个库，分别是`import pytesseract`和`from PIL import Image`。然后OCR识别需要传入一个二值图像，具体的操作代码如下：
```python
#将待检测的二值图像保存成文件
filename = 'buffpic.jpg'
cv2.imwrite(filename, img)

#ocr识别
buf = Image.open(filename)
text = pytesseract.image_to_string(buf)
print(text)
os.remove(filename)
```