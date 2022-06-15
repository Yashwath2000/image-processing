# image-processing
http://localhost:8888/notebooks/yashwath%20ip%20lab/exercises.ipynb

1.develop a program to display greyscale image using read and write operation
import cv2
img1=cv2.imread('flower4.jpg',0)
cv2.imshow('flower4',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/98145017/173816666-6895a466-f0d4-452c-bde7-7352e48c8db9.png)
2.Develop a program to display the image using matplot.lib
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('butterfly2.jpg')
plt.imshow(img)
![image](https://user-images.githubusercontent.com/98145017/173817349-b1c8b2ca-6353-4499-80e4-5a06ac5bb98d.png)
