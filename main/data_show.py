import matplotlib.pyplot as plt
import cv2
from main import data_set as ds
from math import ceil

# 可视化
#展示结果数据，柱状图
def show_result_data():
    pass

#展示特征数据，散点图
def show_feature_data():
    pass

#展示图片分类结果
def show_result_image():
    index=1
    col=10
    for name, count in ds.testset_info.items():
        dir = "../data/test_set/" + name + "/"
        print('start show classified image')
        row = ceil(count / col)
        fig = plt.figure(name)
        for i in range(1, count + 1):
            filename = dir + name + " (" + str(i) + ").jpg"
            img = cv2.imread(filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax = fig.add_subplot(row,col,i)
            ax.imshow(img_gray, cmap="gray")  # 以灰度图显示图片
            # ax.set_title("hei,i'am the second"+filename)  # 给图片加titile
            plt.axis("off")#不显示刻度
            index+=1
        plt.show()  # 显示刚才所画的所有操作

show_result_image()





