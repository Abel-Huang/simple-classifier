# -*- coding=UTF-8 -*-
import cv2
from matplotlib import pyplot as plt

if __name__ == "__main__":
    i=2
    img = cv2.imread('H:\data\image\ls1.bmp', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax.set_title("hei,i'am the first")

    ax = fig.add_subplot(2,2,2)
    ax.imshow(img_gray, cmap="gray")  # 以灰度图显示图片
    ax.set_title("hei,i'am the second")  # 给图片加titile

    ax = fig.add_subplot(2,2,4)
    ax.imshow(img_gray, cmap="gray")  # 以灰度图显示图片
    ax.set_title("hei,i'am the third")  # 给图片加titile
    # plt.axis("off")#不显示刻度
    plt.show()  # 显示刚才所画的所有操作