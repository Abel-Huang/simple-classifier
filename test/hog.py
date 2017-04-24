from cv2 import *
from numpy import *

img = imread('H:\data\image\ls1.bmp', IMREAD_COLOR)
hog = HOGDescriptor()
h = hog.compute(img)
print(type(h))
print(h)