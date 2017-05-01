import cv2

img = cv2.imread('../data/train_set/car/car(12).jpg')
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp,img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift', img)
print(des)
print(type(des))
print(des.shape)
cv2.waitKey(0)