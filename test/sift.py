import cv2
# 读取图片
img = cv2.imread('../data/train_set/car/car(12).jpg')
#转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#构建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()
#获取关键点和位置
kp, des = sift.detectAndCompute(gray, None)
# 绘制SIFT特征点图像
img = cv2.drawKeypoints(gray, kp,img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 显示图像
cv2.imshow('sift', img)
cv2.waitKey(0)