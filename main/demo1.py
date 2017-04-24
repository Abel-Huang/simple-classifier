import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

print(cv2.__version__)
# TrainSetInfo={
#     'Lace': 75,
#     'Printed': 114,
#     'Yarn_dyed': 197
# }
#
# TestSetInfo={
#     'Lace': 75,
#     'Printed': 114,
#     'Yarn_dyed': 197
# }

trainset_info = {
	"car"		:	40,
	"city"		:	20,
	"dog"		:	30,
	"earth"		:	15,
	"fireworks"	:	20,
	"flowers"	:	20,
	"fruits"	:	20,
	"glass"		:	20,
	"gold"		:	15,
	"gun"		:	20,
	"plane"		:	40,
	"sky"		:	30,
	"worldcup"	:	40
}

testset_info = {
	"car"		:	119,
	"city"		:	59,
	"dog"		:	49,
	"earth"		:	24,
	"fireworks"	:	54,
	"flowers"	:	63,
	"fruits"	:	78,
	"glass"		:	52,
	"gold"		:	44,
	"gun"		:	44,
	"plane"		:	102,
	"sky"		:	78,
	"worldcup"	:	131
}

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
    for name, count in trainset_info.items():
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
    for name, count in trainset_info.items():
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


def trainClassifier():
    trainData = np.float32([]).reshape(0, 50)
    response = np.float32([])
    dictIdx = 0
    for name, count in trainset_info.items():
        dir = "../data/test_set/" + name + "/"
        labels, centers = np.load("../data/temp/vocabulary/" + name + ".npy")
        print("Init training data of " + name + "...")
        for i in range(1, count + 1):
            filename = dir + name + " (" + str(i) + ").jpg"
            img = cv2.imread(filename)
            features = calSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)
        res = np.repeat(np.float32([dictIdx]), count)
        response = np.append(response, res)
        dictIdx += 1
        print("Done\n")
    print("Now train svm classifier...")
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    print('trainData \n')
    print(trainData)
    print('response \n')
    print(response)

#  openCV中的SVM
#       svm = cv2.svm
#     params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1)
#     svm.train(trainData, response, params = params)  # select best params
#       svm.save("svm.clf")
#     print("Done\n")

#  sklearn中的SVM
    h = .02  # step size in the mesh
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(trainData, response)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(trainData, response)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(trainData, response)
    lin_svc = svm.LinearSVC(C=C).fit(trainData, response)

    # 保存训练好的模型
    # os.chdir('model_save')
    joblib.dump(svc, "../data/model/svc_model.m")
    joblib.dump(rbf_svc, "../data/model/rbf_svc_model.m")
    joblib.dump(poly_svc, "../data/model/poly_svc_model.m")
    joblib.dump(lin_svc, "../data/model/lin_svc_model.m")


def classify():
    #openCV中SVM
    # svm = cv2.SVM()
    # svm.load("svm.clf")
    #sklearn中的SVM
    # 载入分类器
    svc = joblib.load("../data/model/svc_model.m")
    total = 0; #总量
    correct = 0; #正确分类的总量
    dictIdx = 0 #索引
    for name, count in testset_info.items():
        crt = 0
        dir = "../data/test_set/" + name + "/"
        labels, centers = np.load("../data/temp/vocabulary/" + name + ".npy")
        print("Classify on test_set " + name + ":")
        for i in range(1, count + 1):
            #对每一张图片进行预测
            filename = dir + name + " (" + str(i) + ").jpg"
            img = cv2.imread(filename)
            features = calSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            if (dictIdx == svc.predict(case)):
                log=filename+': is in this class'
                print(log)
                crt += 1
            else:
                log = filename + ': is not in this class'
                print(log)

            # if (dictIdx == svm.predict(case)):
            #     crt += 1
        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1
    print("Total accuracy: " + str(correct) + " / " + str(total))

if __name__ == "__main__":
    initFeatureSet()
    learnVocabulary()
    trainClassifier()
    classify()





