import matplotlib.pyplot as plt
import cv2
from main import data_set as ds
from math import ceil
import numpy as np

# 可视化
#展示结果数据，柱状图，这个是以
def show_result_data(list1, list2, list3, list4, name, llabel):
    n_groups = 12

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22  # 柱条宽度
    opacity = 0.4
    rects1 = plt.bar(index, list1, bar_width, alpha=opacity, color='b',
                     label=llabel[0])  # 第一个柱条，data1调用数据名字，label右上角图标名字
    rects2 = plt.bar(index + bar_width, list2, bar_width, alpha=opacity, color='r', label=llabel[1])
    rects3 = plt.bar(index + 2 * bar_width, list3, bar_width, alpha=opacity, color='g', label=llabel[2])
    rects4 = plt.bar(index + 3 * bar_width, list4, bar_width, alpha=opacity, color='y', label=llabel[3])

    plt.xlabel('Group')
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(index + bar_width, name)
    plt.ylim(0, 100)  # y轴的范围
    plt.legend()

    plt.tight_layout()
    plt.show()

#展示最终统计结果
def show_summary_data(list_1, list_2, list_3, list_4, name, llabel):
    n_groups = 4

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22  # 柱条宽度
    opacity = 0.4
    rects1 = plt.bar(index, list_1, bar_width, alpha=opacity, color='b',label=llabel[0])  # 第一个柱条，data1调用数据名字，label右上角图标名字
    rects2 = plt.bar(index + bar_width, list_2, bar_width, alpha=opacity, color='r', label=llabel[1])
    rects3 = plt.bar(index + 2 * bar_width, list_3, bar_width, alpha=opacity, color='g', label=llabel[2])
    rects4 = plt.bar(index + 3 * bar_width, list_4, bar_width, alpha=opacity, color='y', label=llabel[3])

    plt.xlabel('Group')
    plt.ylabel('Classify Accuracy(%)')
    plt.title('4 feature and 4 kernel histogram')
    plt.xticks(index + bar_width, name)
    plt.ylim(0, 100)  # y轴的范围
    plt.legend()

    plt.tight_layout()
    plt.show()

#展示图片分类结果
def show_result_image():
    index=1
    col=10
    for name, count in ds.testset_info.items():
        dir = '../data/test_set/' + name + '/'
        print('start show classified image')
        row = ceil(count / col)
        fig = plt.figure(name)
        for i in range(1, count + 1):
            filename = dir + name + ' (' + str(i) + ').jpg'
            img = cv2.imread(filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax = fig.add_subplot(row,col,i)
            ax.imshow(img_gray, cmap='gray')  # 以灰度图显示图片
            # ax.set_title("hei,i'am the second"+filename)  # 给图片加titile
            plt.axis('off')#不显示刻度
            index+=1
        plt.show()  # 显示刚才所画的所有操作







