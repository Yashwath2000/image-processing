# image-processing
http://localhost:8888/notebooks/yashwath%20ip%20lab/exercises.ipynb

1.Develop a program to display greyscale image using read and write operation<br>
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
3.Develop a program to perform linear transformation rotation<br>
import cv2<br>
from PIL import Image<br>
img=Image.open("leaf1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/174046198-f567785b-8ee7-4e28-b2b3-fe636624da5b.png)<br>

4.Develop a program to convert color string to RGB color values.<br>
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
5.Write a program to create image using colors <br>
from PIL import Image<br>
img=Image.new("RGB",(200,400),(255,255,0))<br>
img.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/98145017/174035798-b3c298d4-fd20-43ac-8ca9-33787ce7c99e.png)<br>
6.Develop a program to visualise the image using various color spaces<br>
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
7.Write a program to display the image attribute<br>
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
8. Convert the original image to gray scale and then to binary.<br>
import cv2<br>
img=cv2.imread("butterfly2.jpg")<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
#gray scale<br>
img=cv2.imread("butterfly2.jpg",0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
#Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/174050312-ab25033a-034a-42e4-8fee-5d0606f538e1.png)<br>
![image](https://user-images.githubusercontent.com/98145017/174050442-50bd49aa-1d90-439e-a85e-a1b8eb029a5b.png)<br>
![image](https://user-images.githubusercontent.com/98145017/174050511-2fc3f47e-cdbd-466b-ad54-4dd777b1ef80.png)<br>
9.Resize the original image.<br>
import cv2<br>
img=cv2.imread("butterfly3.jpg")<br>
print("original image length width",img.shape)<br>
cv2.imshow("original image",img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow("resized image",imgresize)<br>
print("resized image length width",imgresize.shape)<br>
cv2.waitKey(0)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/174050855-469b2a1d-1d2b-46b4-b498-514b22e00f04.png)<br>
original image length width (558, 1092, 3)<br>
10.Develop a program to readimage using URL
from skimage import io
import matplotlib.pyplot as plt
url="https://images.thequint.com/thequint/2018-01/5d369107-8477-4216-a39d-ad806e1d3a0c/Virat-century.jpg?rect=0%2C0%2C4650%2C2616&auto=format%2Ccompress&fmt=webp.jpg"
image=io.imread(url)
plt.imshow(image)
plt.show()
OUTPUT:
![image](https://user-images.githubusercontent.com/98145017/175017290-507ca0c9-f1b5-43e3-8f8d-59bd7e453877.png)
11. Write a program to mask and blur the image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=cv2.imread('butterfly5.jpg')
plt.imshow(img)
plt.show()
![image](https://user-images.githubusercontent.com/98145017/175017685-3d81a8dd-1c82-49d4-917c-1b458ea73cc3.png)
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(2,1,1)
plt.imshow(mask,cmap="gray")
plt.subplot(2,1,2)
plt.imshow(result)
plt.show()
![image](https://user-images.githubusercontent.com/98145017/175017831-32f9f93e-6de8-4b00-8254-f1eb7a3b5a34.png)
light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
![image](https://user-images.githubusercontent.com/98145017/175017920-0c6af32f-d404-4408-8cd6-92ef5fb72a60.png)
final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()
![image](https://user-images.githubusercontent.com/98145017/175018041-1c4aa58b-15be-4c0a-8fe1-2c5b7a299ac1.png)
blur=cv2.GaussianBlur(final_result,(7,7),0)
plt.imshow(blur)
plt.show()
![image](https://user-images.githubusercontent.com/98145017/175018188-0407af3c-a4f4-43a0-9966-15414e5ad409.png)
13. Write a program to perform arithmatic operations on images
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img1=cv2.imread("image1.jpg")
img2=cv2.imread("image1.jpg")
fimg1=img1 + img2
plt.imshow(fimg1)
plt.show()
cv2.imwrite("output.jpg",fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()

cv2.imwrite("output.jpg",fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()

cv2.imwrite("output.jpg",fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()

cv2.imwrite('output.jpg',fimg4)
OUTPUT:
![image](https://user-images.githubusercontent.com/98145017/175018411-b94ba631-c919-4d32-bec9-60af8e042d74.png)
![image](https://user-images.githubusercontent.com/98145017/175018456-70c20a89-601f-47e2-a455-5ecf7d630cdf.png)
![image](https://user-images.githubusercontent.com/98145017/175018513-2c4372ac-370c-4228-b11d-32403aa82681.png)
![image](https://user-images.githubusercontent.com/98145017/175018608-49625801-289c-4e6a-90cf-fa3785d48a4a.png)












