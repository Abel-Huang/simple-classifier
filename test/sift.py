from cv2 import *
from numpy import *

# import pdb
# pdb.set_trace()#turn on the pdb prompt

# read image
img = imread('H:\data\image\ls1.bmp', IMREAD_COLOR)
gray = cvtColor(img, COLOR_BGR2GRAY)
imshow('origin', img);

# SIFT
detector = xfeatures2d.SURF_create(20)
keypoints = detector.detect(gray, None)
img = drawKeypoints(gray, keypoints,img)
#img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imshow('test', img);
print(img)
waitKey(0)
#destroyAllWindows()