import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('H:\data\image\lena.jpg')
plt.imshow(img)

brisk = cv2.BRISK_create()

 # 计算特征点并显示
(kpt, desc) = brisk.detectAndCompute(img, None)
bk_img = img.copy()
out_img = img.copy()
out_img = cv2.drawKeypoints(bk_img, kpt, out_img)
plt.figure(2)
plt.imshow(out_img)

# 原图像旋转30度
ang = np.pi / 6
rot_mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 200]])
img_30 = cv2.warpAffine(img, rot_mat, (600, 500))
plt.figure(3)
plt.imshow(img_30)

# 特征点检测
(kpt_30, desc_30) = brisk.detectAndCompute(img_30, None)
bk_img = img_30.copy()
out_img = img_30.copy()
out_img = cv2.drawKeypoints(bk_img, kpt_30, out_img)
plt.figure(4)
plt.imshow(out_img)

# 特征点匹配
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc, desc_30)
print(matches)

# 显示匹配结果，仅显示前面的5个点
matches=sorted(matches, key = lambda x:x.distance)
out_img = cv2.drawMatches(img, kpt, img_30, kpt_30, matches[0:5], out_img)
plt.figure(5)
plt.imshow(out_img)
plt.show()