from cv2 import *
import sklearn
#read image
img = imread('H:\data\image\ls1.bmp', IMREAD_COLOR)
gray = cvtColor(img, COLOR_BGR2GRAY)
