# image-processing
http://localhost:8888/notebooks/yashwath%20ip%20lab/exercises.ipynb

1.develop a program to display greyscale image using read and write operation
import cv2
img1=cv2.imread('flower4.jpg',0)
cv2.imshow('flower4',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/98145017/174034372-bacd8bc6-134a-4167-a42c-d2aacab27b06.png)<br>

2.Develop a program to display the image using matplot.lib
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('butterfly2.jpg')
plt.imshow(img)
![image](https://user-images.githubusercontent.com/98145017/173817349-b1c8b2ca-6353-4499-80e4-5a06ac5bb98d.png)
3.
import cv2
from PIL import Image
img=Image.open("leaf1.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
4.
import cv2
from PIL import ImageColor
#using getrgb for yellow
img1=ImageColor.getrgb("yellow")
print(img1)
#using getrgb for red
img2=ImageColor.getrgb("red")
print(img2)
output:
(255, 255, 0)
(255, 0, 0)
5.
from PIL import Image
img=Image.new("RGB",(200,400),(255,255,0))
img.show()
output:
![image](https://user-images.githubusercontent.com/98145017/174035798-b3c298d4-fd20-43ac-8ca9-33787ce7c99e.png)
6.
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread("leaf1.jpg")
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()
OUTPUT:
![image](https://user-images.githubusercontent.com/98145017/174035974-b0cfe492-6945-417c-ba0e-02762aaf1964.png)
![image](https://user-images.githubusercontent.com/98145017/174036076-ea7335f1-103b-4f4f-8858-18d16d193499.png)
7.
from PIL import Image
image=Image.open("leaf1.jpg")
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()
OUTPUT:
Filename: leaf1.jpg
Format: JPEG
Mode: RGB
Size: (4344, 2896)
Width: 4344
Height: 2896
