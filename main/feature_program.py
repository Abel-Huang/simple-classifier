import cv2
import numpy as np
from main import data_set as ds

print(cv2.__version__)

#计算sift特征

def calSiftFeature(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create(200)
    kp, des = sift.detectAndCompute(gray, None)
    return des

#计算特征向量

def  calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):#获取矩阵的维数
		fi = features[i]
		diffMat = np.tile(fi, (50, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()#返回的是数组值从小到大的索引值
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1
	return featVec


# 初始化特征集
def initFeatureSet():
    for name, count in ds.trainset_info.items():
        dir = "../data/train_set/" + name + "/"
        featureSet = np.float32([]).reshape(0, 128)
        print("Extract features from training set" + name + "...")
        for i in range(1, count + 1):
            filename = dir + name + " (" + str(i) + ").jpg"
            img = cv2.imread(filename)
            des = calSiftFeature(img)
            featureSet = np.append(featureSet, des, axis=0)
        featCnt = featureSet.shape[0]
        print(str(featCnt) + " features in " + str(count) + " images\n")
        # save featureSet to file
        filename = "../data/temp/features/" + name + ".npy"
        np.save(filename, featureSet)


def learnVocabulary():
    wordCnt = 50
    for name, count in ds.trainset_info.items():
        filename = "../data/temp/features/" + name + ".npy"
        features = np.load(filename)
        print("Learn vocabulary of " + name + "...")
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
        compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)
        # save vocabulary(a tuple of (labels, centers)) to file
        filename = "../data/temp/vocabulary/" + name + ".npy"
        np.save(filename, (labels, centers))
        print("Done\n")








