# image-processing
http://localhost:8888/notebooks/yashwath%20ip%20lab/exercises.ipynb

1.develop a program to display greyscale image using read and write operation<br>
import cv2<br>
img1=cv2.imread('flower4.jpg',0)<br>
cv2.imshow('flower4',img1)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145017/174034372-bacd8bc6-134a-4167-a42c-d2aacab27b06.png)<br>

2.Develop a program to display the image using matplot.lib<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('butterfly2.jpg')<br>
plt.imshow(img)<br>
![image](https://user-images.githubusercontent.com/98145017/173817349-b1c8b2ca-6353-4499-80e4-5a06ac5bb98d.png)<br>
3.
import cv2<br>
from PIL import Image<br>
img=Image.open("leaf1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
4.<br>
import cv2<br>
from PIL import ImageColor<br>
#using getrgb for yellow<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
#using getrgb for red<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
output:<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
5.<br>
from PIL import Image<br>
img=Image.new("RGB",(200,400),(255,255,0))<br>
img.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/98145017/174035798-b3c298d4-fd20-43ac-8ca9-33787ce7c99e.png)<br>
6.<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread("leaf1.jpg")<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/174035974-b0cfe492-6945-417c-ba0e-02762aaf1964.png)<br>
![image](https://user-images.githubusercontent.com/98145017/174036076-ea7335f1-103b-4f4f-8858-18d16d193499.png)<br>
7.<br>
from PIL import Image<br>
image=Image.open("leaf1.jpg")<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>
OUTPUT:<br>
Filename: leaf1.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (4344, 2896)<br>
Width: 4344<br>
Height: 2896<br>
