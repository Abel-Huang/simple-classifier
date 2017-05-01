import cv2

img = cv2.imread('H:\data\image\ls1.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('surf', img)
print(des)
print(type(des))
print(des.shape)
cv2.waitKey(0)
