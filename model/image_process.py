'''
夜间近红外图像增强
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

orimage_infrared='images/infrared.jpg'
orimage_visual='images/visual.jpg'
# Gamma增强
def gama_transfer(img,power1):
    if len(img.shape) == 3:
         img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = 255*np.power(img/255,power1)
    img = np.around(img)
    img[img>255] = 255
    out_img = img.astype(np.uint8)
    return out_img

img1 = cv2.imread(orimage_infrared)
img2=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

img2_dst = cv2.blur(img2, (30, 30))#均值滤波
img2_Guassian = cv2.GaussianBlur(img2,(9,9),0)

img2_gamma=gama_transfer(img2_Guassian,0.6)# Gamma增强
cv2.imwrite('infrared_enforce.jpg',img2_gamma)

img1=cv2.resize(img1,(600,400))
img2=cv2.resize(img2,(600,400))
img2_gamma=cv2.resize(img2_gamma,(600,400))
img2_dst=cv2.resize(img2_dst,(600,400))

cv2.imshow('orimage',img2)
cv2.imshow('processed',img2_gamma)

cv2.waitKey(0)