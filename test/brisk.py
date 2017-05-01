import cv2

img = cv2.imread('../data/train_set/sky/sky(1).jpg')
cv2.imshow('hehe',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brisk = cv2.BRISK_create()
kp, des = brisk.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp, (255, 0, 0))
cv2.imshow('brisk', img)
print(des)
print(type(des))
print(des.shape)
cv2.waitKey(0)
