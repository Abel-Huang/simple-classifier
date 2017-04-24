from cv2 import *
import numpy as np
from matplotlib import pyplot as plt


def fun1():
    img = imread('H:\data\image\ls1.bmp', IMREAD_GRAYSCALE)
    # bins->图像中分为多少格；range->图像中数字范围
    plt.hist(img.ravel(), bins=256, range=[0, 256]);
    plt.show()


def fun2():
    img = imread('H:\data\image\ls1.bmp', IMREAD_COLOR)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.show()
fun1()
fun2()