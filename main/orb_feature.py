import cv2
import numpy as np
from main import data_set as ds

# 计算ORB特征

def cal_orb_feature(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 最大特征点数
    orb = cv2.ORB_create(50000)
    kp, des = orb.detectAndCompute(gray, None)
    return des

# 初始化特征集
def init_feature_set():
    for name, count in ds.trainset_info.items():
        dir = '../data/train_set/' + name + '/'
        # 设置feature_set的维度为0行32列 因为des返回的是32
        feature_set = np.float32([]).reshape(0, 32)
        print('Extract features from training set' + name + '...')
        for i in range(1, count + 1):
            filename = dir + name + ' (' + str(i) + ').jpg'
            img = cv2.imread(filename)
            des = cal_orb_feature(img)
            print(des.shape)
            feature_set = np.append(feature_set, des, axis=0)
        feat_cnt = feature_set.shape[0]
        print(str(feat_cnt) + ' features in ' + str(count) + ' images\n')
        # save featureSet to file
        filename = '../data/temp/orb/features/' + name + '.npy'
        np.save(filename, feature_set)

# 利用k-means获取标签
def learn_vocabulary():
    word_cnt = 50
    for name, count in ds.trainset_info.items():
        filename = '../data/temp/orb/features/' + name + '.npy'
        features = np.load(filename)
        print('Learn vocabulary of ' + name + '...')
        # use k-means to cluster a bag of features
        # —–cv2.TERM_CRITERIA_EPS:精确度（误差）满足epsilon停止。
        # —- cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
        # —-cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        #分类数据, 分类个数, 预设的分类标签(没有的话 None),
        # 迭代停止的模式选择,重复试验kmeans算法次数，将会返回最好的一次结果
        # 初始类中心选择，两种方法
        compactness, labels, centers = cv2.kmeans(features, word_cnt, None, criteria, 20, flags)
        # save vocabulary(a tuple of (labels, centers)) to file
        filename = '../data/temp/orb/vocabulary/' + name + '.npy'
        np.save(filename, (labels, centers))
        print('Done\n')

init_feature_set()
learn_vocabulary()

