import cv2

img1 = cv2.imread('H:\data\image\ls1.bmp')
img2 = cv2.imread('H:\data\image\qw.jpg')

gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 最大特征点数
orb = cv2.ORB_create(50000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img1 = cv2.drawKeypoints(gray1, kp1, img1)
img2 = cv2.drawKeypoints(gray2, kp2, img2)

cv2.imshow('orb1', img1)
cv2.imshow('orb2', img2)

print(des1)
print(type(des1))
print(des1.shape)
print(des2.shape)
cv2.waitKey(0)

# # 提取并计算特征点
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# # knn筛选结果
# matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
# good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
# # 查看最大匹配点数目
# print(len(good))