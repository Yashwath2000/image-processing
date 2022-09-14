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
10.Develop a program to readimage using URL<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url="https://images.thequint.com/thequint/2018-01/5d369107-8477-4216-a39d-ad806e1d3a0c/Virat-century.jpg?rect=0%2C0%2C4650%2C2616&auto=format%2Ccompress&fmt=webp.jpg"<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/175017290-507ca0c9-f1b5-43e3-8f8d-59bd7e453877.png)<br>
11. Write a program to mask and blur the image<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('butterfly5.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/175017685-3d81a8dd-1c82-49d4-917c-1b458ea73cc3.png)<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(2,1,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(2,1,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/175017831-32f9f93e-6de8-4b00-8254-f1eb7a3b5a34.png)<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/175017920-0c6af32f-d404-4408-8cd6-92ef5fb72a60.png)<br>
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)
plt.imshow(final_result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/175018041-1c4aa58b-15be-4c0a-8fe1-2c5b7a299ac1.png)<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/175018188-0407af3c-a4f4-43a0-9966-15414e5ad409.png)<br>
13. Write a program to perform arithmatic operations on images<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread("image1.jpg")<br>
img2=cv2.imread("image1.jpg")<br>
fimg1=img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite("output.jpg",fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
<br>
cv2.imwrite("output.jpg",fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
<br>
cv2.imwrite("output.jpg",fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
<br>
cv2.imwrite('output.jpg',fimg4)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/175018411-b94ba631-c919-4d32-bec9-60af8e042d74.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175018456-70c20a89-601f-47e2-a455-5ecf7d630cdf.png)<br><br>
![image](https://user-images.githubusercontent.com/98145017/175018513-2c4372ac-370c-4228-b11d-32403aa82681.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175018608-49625801-289c-4e6a-90cf-fa3785d48a4a.png)<br>
12.<br>
import cv2 <br>
img=cv2.imread("flower3.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/175282593-176f9f27-04a8-475f-90f2-cd472146359c.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175282653-fb6c7d91-1484-47d4-b0be-0efd966ea055.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175282706-875418a3-10e7-4086-8ca2-c32e0c56d67c.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175282784-8071bfe6-8c54-45d1-a32b-e7ca65d038f9.png)<br>
![image](https://user-images.githubusercontent.com/98145017/175282855-7209bb80-6734-4622-b46b-be770912bfce.png)<br>
13.<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save("image3.png")<br>
img.show()<br>
c.waitKey(0)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/175283161-ed7c91f2-a773-486c-ba24-b5b699f20fe1.png)<br>
exer 2;http://localhost:8888/notebooks/yashwath%20ip%20lab/exercise2.ipynb<br>
erse 3:http://localhost:8888/notebooks/yashwath%20ip%20lab/exercise3.ipynb<br>
exer 4:http://localhost:8888/notebooks/yashwath%20ip%20lab/exercise4.ipynb<br>
14.<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('butterfly1.jpg',1)<br>
image2=cv2.imread('butterfly1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/176421321-a25de709-26e0-48f1-9369-bbfc0dfd95cc.png)<br>
15.<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread("virat2.jpg")<br>
cv2.imshow("Original Image",image)<br>
cv2.waitKey(0)<br>
<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow("Gaussian Blurring",Gaussian)<br>
cv2.waitKey(0)<br>
<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow("median Blurring",median)<br>
cv2.waitKey(0)<br>
<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow("Bilateral Blurring",bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/176421758-3e00521e-9555-4926-8cf5-4d724352faea.png)<br>
![image](https://user-images.githubusercontent.com/98145017/176421861-95b713e3-c896-48ae-ac83-b0dc095eaa7a.png)<br>
![image](https://user-images.githubusercontent.com/98145017/176422009-8455b29b-ea1a-4743-aa04-0bbe0bfa63a5.png)<br>
![image](https://user-images.githubusercontent.com/98145017/176422116-5b2a0c56-777f-4943-a817-31d6bc7c4bf4.png)<br>
16.<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open("butterfly1.jpg")<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/176422670-086bde87-4474-44c7-91b4-c0361df93a13.png)<br>
17.<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('virat1.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel= np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)<br>
closing=cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/176422923-5a986101-0b2d-497f-b64b-dea7ad122afd.png)<br>
18. Develop a program to<br>
(i)                Read the image, convert it into grayscale image<br>
(ii)              write (save) the grayscale image and<br>
(iii)            display the original image and grayscale image<br>
import cv2<br>
OriginalImg=cv2.imread("virat2.jpg")<br>
GrayImg=cv2.imread('virat2.jpg',0)<br>
isSaved=cv2.imwrite('E:/i.jpg',GrayImg)<br>
cv2.imshow('Display Orignal Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('the image is successfully saved.')<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/178713544-a9036a40-0cf3-4c2d-a02a-3f8b1412de08.png)<br>
19.Slicing_With_Background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
             z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/178713947-7e60b8f0-630d-47f6-8603-d3f441d4d748.png)<br>
20.Slicing_Without_Background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
             z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/178714183-660dd3d4-548c-4e10-a607-3ff804b6b760.png)<br>
21.Analyse gthe data using Histogram<br><br>
import numpy as np<br>
import skimage.color<br>
import skimage.io<br>
import matplotlib.pyplot as plt<br>
#%matplotlib widget<br>
<br>
# read the image of a plant seedling as grayscale from the outset<br>
image = skimage.io.imread(fname="virat1.jpg", as_gray=True)<br>
<br>
# display the image<br>
fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>
<br>
# create the histogram<br>
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))<br>
<br>
# configure and draw the histogram figure<br>
plt.figure()<br>
plt.title("Grayscale Histogram")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0, 1.0])  # <- named arguments do not work here<br>
<br>
plt.plot(bin_edges[0:-1], histogram)  # <- or here<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/178972486-ab17b2e1-1768-4463-b1f0-97e2d55ebde1.png)<br>
22. Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>
d)<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('butterfly4.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/179970623-d3ea02f5-6c51-4354-a60c-85484fe92653.png)<br>
negative=255- pic #neg =(L-1)-img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/179971256-ddf1d8ab-58fd-4904-b0eb-eb3f8aeea87d.png)<br>
%matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('butterfly4.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
<br>
max_=np.max(gray)<br>
<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/179971385-6e43bdcb-8b0f-40e6-8616-4686dd17d7b9.png)<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
<br>
# Gamma encoding <br>
pic=imageio.imread('butterfly4.jpg')<br>
gamma=2.2# Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright<br>
<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145017/179971563-0d814c24-ac1a-4f5e-aac7-1a5e15df57df.png)<br>
23. Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>
#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
my_image=Image.open('image1.jpg')<br>
#Use sharpen funcion<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#save the image<br>
sharp.save('E:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/179971757-5c80a6ac-e8f2-4988-af6f-64275f7fd74e.png)<br>
#Image flip<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
img=Image.open('image1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
<br>
#save the image<br>
flip.save('E:/image_flip.jpg')
plt.imshow(flip)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/179971981-1053411e-b8f7-49e8-ba05-974d4bd956b3.png)<br>
#Importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#Opens a image in RGB mode<br>
im=Image.open('image1.jpg')<br>
<br>
#Size of the image in pixels(size of original image)<br>
width,height=im.size<br>
<br>
#cropped image of above dimension<br>
im1=im.crop((1000,500,3000,1750))<br>
<br>
#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145017/179972176-16a4378d-e091-4792-bb7e-ca5669599da3.png)<br>
**program to display the different color in diagonal with matrix<br>
from PIL import Image<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 600, 600<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400,300:400]=[0,0,255]<br>
data[400:500,400:500]=[255,255,0]<br>
data[500:600,500:600]=[0,255,255]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
output;<br>
![image](https://user-images.githubusercontent.com/98145017/186405539-e7fff60b-e2d0-4df6-8d05-72f1cd420a3b.png)<br>
![image](https://user-images.githubusercontent.com/98145017/186405658-a1529044-e359-41bd-ae4a-6c15f91bbf33.png)<br>
<br>
31.Read an image to find max,min,average and standard deviation of pixel value.<br>
from numpy import asarray<br>
from PIL import Image<br>
image = Image.open('cat1.jpg')<br>
pixels = asarray(image)<br>
#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
pixels = pixels.astype('float32')<br>
pixels /= 255.0<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("cat1.jpg",0)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("cat1.jpg",0)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>
from PIL import Image,ImageStat<br><br>
import matplotlib.pyplot as plt<br>
im=Image.open('cat1.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('cat1.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
print(max_channels)<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('cat1.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
print(min_channels)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/186652297-9b18dcff-449d-4923-baf3-6ff0c9eca037.png)<br>
![image](https://user-images.githubusercontent.com/98145017/186652390-fe22537d-330d-4aa1-84e5-9bfbb97e7dc8.png)<br>
![image](https://user-images.githubusercontent.com/98145017/186652493-8bda717d-9b7c-4c26-b831-52e85013492b.png)<br>
######<br>
def printPattern(n):<br>
    arraySize = n * 2 - 1;
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
    #Fill the values<br><br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
    #Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 #Driver Code<br>
n = 4;<br>
printPattern(n);<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/186652645-3f726d59-1f1b-4e52-893c-d7f7d0f07ef1.png)<br>
<br>
**Sobel edge and canny edge detection<br>
import cv2<br>
# Read the original image<br>
img = cv2.imread('bigbull.jpg')<br>
# Display original imag<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>
# Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
# Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>
# Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
# Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
# Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
# Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/98145017/186653583-5990aa0f-493f-4601-a834-50fc842343d3.png)<br>

Basic pillow functions<br>
A.<br>
from PIL import Image, ImageChops, ImageFilter<br>
from matplotlib import pyplot as plt<br>
<br>
x = Image.open("x.png")<br>
o=Image.open("o.png")<br>

print('size of the image:', x.size, 'colour mode:', x.mode)<br>
print('size of the image: ', o.size, 'colour mode:', o.mode)<br>
<br>
plt.subplot(121),plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>

merged=ImageChops.multiply(x,o)<br>
add=ImageChops.add(x,o)<br>

greyscale=merged.convert('L')<br>
greyscale<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187899655-dc72a406-ad55-407a-9fac-7001ac770153.png)<br>
B.<br>
image=merged<br>
print('image size:',image.size,<br>
'\ncolor mode:', image.mode,<br>
'\nimage width:', image.width,'| also represented by:',image.size[0],<br>
'\nimage height:', image.height, '| also represented by:',image.size[1],)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187899821-29890323-bf65-4d25-a9df-fdc3d6ed0a72.png)<br>
C.<br>
pixel = greyscale.load()<br>
for row in range (greyscale.size[0]):<br>
 for column in range(greyscale.size[1]):<br>
    if pixel[row, column] != (255):<br>
     pixel[row, column] = (0)<br>
greyscale<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187900095-13368319-1611-47bd-94c3-f1abddab9626.png)<br>
D.<br>
invert = ImageChops.invert(greyscale)<br>
<br>
bg=Image.new('L', (256, 256), color=(255))<br>
subt=ImageChops. subtract (bg, greyscale)<br>
rotate =subt.rotate(45)<br>
rotate<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187900432-33b6af23-5391-4d18-9e5a-4a6749a2f265.png)<br>
E.<br>
blur=greyscale.filter(ImageFilter.GaussianBlur (radius=1))<br>
edge=blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187900676-e10c2344-40ad-4649-add6-03fadc19c321.png)<br>
F.<br>
edge=edge.convert('RGB')<br>
<br>
bg_red=Image.new('RGB', (256,256), color=(255,0,0))<br>
filled_edge = ImageChops.darker(bg_red, edge)<br>
filled_edge<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187900778-9c07b44a-8ec5-42bc-9a0b-fa425e13fd09.png)<br>
<br>
Image restoration<br>
a)restore damaged images<br>
import cv2<br>
import numpy as np
import matplotlib.pyplot as plt<br>
#open the image<br>
img=cv2.imread('billi.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#load the image<br>
mask=cv2.imread('mask.jpg',0)<br>
plt.imshow(mask)<br>
plt.show()<br>
#inpaint<br>
dst=cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)<br>
<br>
#write the output<br>
cv2.imwrite('output1.jpg',dst)<br>
plt.imshow(dst)<br>
plt.show()  <br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145017/187903188-91f5a34e-a046-4354-975d-0d8d3a4626c0.png)<br>
![image](https://user-images.githubusercontent.com/98145017/187903273-2776b060-cee1-4bbd-bd72-3de75c5646ac.png)<br>
![image](https://user-images.githubusercontent.com/98145017/187903411-b3ce64bb-c95e-4e3f-b9b2-24ab99b9b4a6.png)<br>
b)Removing logos<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
plt.rcParams['figure.figsize']=(10,8)<br>
<br>
def show_image(image,tittle='Image', cmap_type='gray'):<br>
    plt.imshow(image,cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
    <br>
def plot_comparison(img_original, img_filtered,img_title_filtered):<br>
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,8), sharex=True, sharey=True)<br>
    ax1.imshow(img_original,cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered, cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
    <br>
from skimage.restoration import inpaint<br>
from skimage.transform import resize<br>
from skimage import color<br>
<br>
image_with_logo=plt.imread('imlogo.png')<br>
#initialise the mask<br>
mask=np.zeros(image_with_logo.shape[:-1])<br>
<br>
#set the [pixels where the logo is to 1<br>
mask[210:272,360:425]=1<br>
<br>
#apply inpainting to remove the logo<br>
image_logo_removed=inpaint.inpaint_biharmonic(image_with_logo,mask,multichannel=True)<br>
<br>
#show the originaland logo removed images<br>
plot_comparison(image_with_logo,image_logo_removed,"Image with logo removed")<br>
![image](https://user-images.githubusercontent.com/98145017/187903750-7c8d9a2f-2722-4ece-bd32-4fc223b9c96e.png)<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
<br>
<br>
from skimage.util import random_noise<br>
fruit_image=plt.imread('fruitts.jpeg')<br>
<br>
#Add noise to the image<br>
noisy_image=random_noise(fruit_image)<br>
<br>
#show th original and resulting image<br>
plot_comparison(fruit_image,noisy_image,'Noisy_image')<br>
![image](https://user-images.githubusercontent.com/98145017/187904388-43cf3ac6-8cea-4b3f-9d80-2a9ca2203a52.png)<br>
from skimage.restoration import denoise_tv_chambolle<br>
<br>
noisy_image=plt.imread('noisy.jpg')<br>
<br>
#Apply total variation filtern denoising<br>
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)<br>
<br>
#show the noisy and denoised image<br>
plot_comparison(noisy_image,denoised_image,'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/98145017/187904441-6e1e556d-90bb-4978-8f47-1a31b896d9ce.png)<br>
from skimage.restoration import denoise_bilateral<br>
<br>
landscape_image= plt.imread('noisy.jpg')<br>
<br>
#Apply bilateral filletr denoising<br>
denoised_image=denoise_bilateral(landscape_image,multichannel=True)<br>
<br>
#show original and resulting images<br>
plot_comparison(landscape_image,denoised_image,'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/98145017/187904514-ca34a450-1c6c-4bde-a330-e9f37a7062c5.png)<br>
#Segmentation:<br>

from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
<br>
face_image=plt.imread('face.jpg')<br>
<br>
#obtain the segmentation with 400 regions<br>
segments=slic(face_image,n_segments=400)<br>
<br>
#put segments on top of original image to compare<br>
segmented_image= label2rgb(segments,face_image,kind='avg')<br>
<br>
#show the segmented image<br>
plot_comparison(face_image,segmented_image,'segmented image,400 superpixels')<br>
![image](https://user-images.githubusercontent.com/98145017/187904573-b4f7fbd2-c715-42db-b73c-5e6a57cbb687.png)<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
plt.rcParams['figure.figsize']=(10,8)<br>
def show_image_contour(image, contours):<br>
    plt.figure()<br>
    for n, contour in enumerate (contours):<br>
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)<br>
    plt.imshow(image, interpolation='nearest', cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
<br>
from skimage import measure, data<br>
horse_image = data.horse()<br>
contours = measure.find_contours(horse_image, level=0.8)<br>
show_image_contour(horse_image, contours)<br>
![image](https://user-images.githubusercontent.com/98145017/187904694-67a1a000-053e-45b5-9aa2-d1cf1cbf592e.png)<br>
from skimage.restoration import inpaint<br>
from skimage.transform import resize<br>
from skimage import color<br>
from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>
image_dices = imread('diceimg.png')<br>
image_dices = color.rgb2gray(image_dices)<br>
thresh = threshold_otsu(image_dices)<br>
binary = image_dices > thresh<br>
contours = measure.find_contours(binary, level=0.8)<br>
show_image_contour(image_dices,contours)<br>
![image](https://user-images.githubusercontent.com/98145017/187904782-8fad952d-a08b-4632-b5f2-0c58733b2672.png)<br>
import numpy as np<br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>
<br>
max_dots_shape = 50<br>
<br>
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]<br>
<br>
show_image_contour (binary, contours)<br>
<br>
print('Dices dots number:{}.'.format(len(dots_contours)))<br>
![image](https://user-images.githubusercontent.com/98145017/187904836-f86b4b33-4add-4008-8510-4a712923f60b.png)<br>
<br>
**36.Implement a program to perform various edge detection techniques**<br>
a) Canny Edge detection<br>
#Canny Edge detection<br>
import cv2<br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
plt.style.use('seaborn')<br>
loaded_image = cv2.imread("animated.jpeg")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br>
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>
plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image,cmap="gray")<br>
plt.title("original Image")<br>
plt.axis("off")<br>
plt.subplot(1,3,2)<br>
plt.imshow(gray_image, cmap="gray")<br>
plt.axis("off")<br>
plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>
plt.title("Canny Edge Detected Image")<br>
plt.show()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187898399-3c66fa2e-2d1b-49bd-9b73-88ce42fdbb28.png)<br>
<br>
b) Edge detection schemes - the gradient (Sobel - first order derivatives) based edge detector and the Laplacian (2nd order derivative, so it is extremely sensitive to noise) based edge detector.<br>
import cv2<br>
import numpy as np <br>
from matplotlib import pyplot as plt<br>
img0=cv2.imread('animated.jpeg',)<br>
gray= cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)<br>
img= cv2.GaussianBlur (gray, (3,3),0)<br>
laplacian= cv2.Laplacian (img,cv2.CV_64F)<br>
sobelx = cv2.Sobel (img,cv2.CV_64F,1,0,ksize=5) <br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) <br>
plt.subplot(2,2,1), plt.imshow(img, cmap = 'gray')<br>
plt.title('Original'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,2), plt.imshow(laplacian,cmap = 'gray') <br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')<br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray')<br>
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])<br>
plt.show()<br>
**OUTPUT:-**
![image](https://user-images.githubusercontent.com/98141711/187899484-5716c42d-9bd2-4a5a-8c3f-0fdf535b965d.png)<br>
c) Edge detection using Prewitt Operator<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
img = cv2.imread('animated.jpeg')<br>
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) <br>
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)<br>
kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])<br>
kernely=np.array([[-1,0,1], [-1,0,1],[-1,0,1]]) <br>
img_prewittx= cv2.filter2D(img_gaussian, -1, kernelx) <br>
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)<br>
cv2.imshow("Original Image", img)<br>
cv2.imshow("Prewitt x", img_prewittx)<br>
cv2.imshow("Prewitt y", img_prewitty)<br>
cv2.imshow("Prewitt", img_prewittx + img_prewitty)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187903451-07fca319-5992-4cd6-aad7-83ee173f7f77.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903539-69d7953e-df4d-4542-8787-228590034af0.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903619-71a5c34a-d089-481e-b424-b8f3bb60b9e0.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903690-a9d951a7-c0f1-442b-b4bd-c03873c5532d.png)<br>
<br>
d) Roberts Edge Detection- Roberts cross operator<br>
import cv2<br>
import numpy as np<br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt <br>
roberts_cross_v = np.array([[1, 0],<br>
                            [0,-1]])<br>
roberts_cross_h= np.array([[0, 1],<br>
                           [-1,0]])<br>
img= cv2.imread("animated.jpeg",0).astype('float64')<br>
img/=255.0<br>
vertical= ndimage.convolve( img, roberts_cross_v)<br>
horizontal=ndimage.convolve( img, roberts_cross_h)<br>
edged_img= np.sqrt( np.square (horizontal) + np.square(vertical))<br>
edged_img*=255<br>
cv2.imwrite("output.jpg",edged_img)<br>
cv2.imshow("OutputImage", edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllwindows()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187904755-c8619d95-810b-49c3-b3bf-e485c8ad7d3a.png)<br>







