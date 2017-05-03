import cv2
import numpy as np
from main import data_set as ds
from util import file_manage as fm
print(cv2.__version__)

#计算特征
def cal_feature_info(img, feature_type):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift=fm.init_feature_obj(feature_type)
    kp, des = sift.detectAndCompute(gray, None)
    return des

#计算特征向量
def  cal_feature_vec(features, centers):
	feat_vec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):#获取矩阵的维数
		fi = features[i]
		diff_mat = np.tile(fi, (50, 1)) - centers
		sq_sum = (diff_mat**2).sum(axis=1)
		dist = sq_sum**0.5
		sorted_indices = dist.argsort()#返回的是数组值从小到大的索引值
		idx = sorted_indices[0] # index of the nearest center
		feat_vec[0][idx] += 1
	return feat_vec

# 初始化特征集
def init_feature_set(feature_type):
    for name, count in ds.trainset_info.items():
        dir = '../data/train_set/' + name + '/'
        # 设置feature_set的维度为0行128列 因为sift des返回的是128
        dimension=fm.init_feature_dimension(feature_type)
        feature_set = np.float32([]).reshape(0, dimension)
        print('Extract features from training set ' + name + '...')
        for i in range(1, count + 1):
            filename = dir + name + ' (' + str(i) + ').jpg'
            img = cv2.imread(filename)
            print(filename)
            des = cal_feature_info(img, feature_type)
            feature_set = np.append(feature_set, des, axis=0)
        feat_cnt = feature_set.shape[0]
        print(str(feat_cnt) + ' features in ' + str(count) + ' images\n')
        # save featureSet to file
        filename = fm.generic_fea_filename(feature_type) + '/features/'+name + '.npy'
        np.save(filename, feature_set)

# 利用k-means获取标签
def learn_vocabulary(feature_type):
    word_cnt = 50
    for name, count in ds.trainset_info.items():
        filename = fm.generic_fea_filename(feature_type) + '/features/'+name + '.npy'
        features = np.load(filename)
        print('Learn vocabulary of ' + name + '...')
        # use k-means to cluster a bag of features
        # —–cv2.TERM_CRITERIA_EPS:精确度（误差）满足epsilon停止。
        # —- cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
        # —-cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        print(type(flags))
        #分类数据, 分类个数, 预设的分类标签(没有的话 None),
        # 迭代停止的模式选择,重复试验kmeans算法次数，将会返回最好的一次结果
        # 初始类中心选择，两种方法
        compactness, labels, centers = cv2.kmeans(features, word_cnt, None, criteria, 20, flags)
        # save vocabulary(a tuple of (labels, centers)) to file
        filename = fm.generic_fea_filename(feature_type) + '/vocabulary/' + name + '.npy'
        np.save(filename, (labels, centers))
        print('Done\n')








