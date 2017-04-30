import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from main import data_set as ds
from main import feature_program as fp
from util import save_2_db as db
from util import create_unique as cu
from util import file_manage as fm

# 训练分类器
def train_classifier(feature_type):
    train_data = np.float32([]).reshape(0, 50)
    response = np.float32([])
    dict_idx = 0
    for name, count in ds.trainset_info.items():
        dir = '../data/test_set/' + name + '/'
        file_name=fm.generic_fea_filename(feature_type) + '/vocabulary/' + name + '.npy'
        labels, centers = np.load(file_name)
        print('Init training data of ' + name + '...')
        for i in range(1, count + 1):
            filename = dir + name + ' (' + str(i) + ').jpg'
            img = cv2.imread(filename)
            features = fp.cal_feature_info(img, feature_type)
            feat_vec = fp.cal_feature_vec(features, centers)
            train_data = np.append(train_data, feat_vec, axis=0)
        res = np.repeat(np.float32([dict_idx]), count)
        response = np.append(response, res)
        dict_idx += 1
        print('Done\n')
    print('Now train svm classifier...')
    train_data = np.float32(train_data)
    response = response.reshape(-1, 1)
    print('trainData \n')
    print(train_data)
    print('response \n')
    print(response)

#  sklearn中的SVM
    h = .02  # step size in the mesh
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(train_data, response)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_data, response)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_data, response)
    lin_svc = svm.LinearSVC(C=C).fit(train_data, response)

    # 保存训练好的模型
    joblib.dump(svc, fm.generic_ml_filename(feature_type, 'svc'))
    joblib.dump(rbf_svc, fm.generic_ml_filename(feature_type, 'rbf'))
    joblib.dump(poly_svc, fm.generic_ml_filename(feature_type, 'poly'))
    joblib.dump(lin_svc, fm.generic_ml_filename(feature_type, 'lin'))

# 调用分类器进行分类
def classify(feature_type, ml_method):
    unitag=cu.create_unique()
    #sklearn中的SVM
    # 载入分类器
    svc = joblib.load(fm.generic_ml_filename(feature_type, ml_method))
    total = 0; #总量
    correct = 0; #正确分类的总量
    dict_idx = 0 #索引
    for name, count in ds.testset_info.items():
        crt = 0
        dir = '../data/test_set/' + name + '/'
        file_name = fm.generic_fea_filename(feature_type) + '/vocabulary/' + name + '.npy'
        labels, centers = np.load(file_name)
        print('Classify on test_set ' + name + ':')
        for i in range(1, count + 1):
            #对每一张图片进行预测
            filename = dir + name + ' (' + str(i) + ').jpg'
            img = cv2.imread(filename)
            features = fp.cal_feature_info(img)
            feat_vec = fp.cal_feature_vec(features, centers)
            case = np.float32(feat_vec)
            if (dict_idx == svc.predict(case)):
                db.store_single(filename, name, ml_method, feature_type, 1, unitag)
                log=filename+': is in this class'
                print(log)
                crt += 1
            else:
                db.store_single(filename, name, ml_method, feature_type, 0, unitag)
                log = filename + ': is not in this class'
                print(log)

        print('Accuracy: ' + str(crt) + ' / ' + str(count) + '\n')
        db.store_total(name, ml_method, feature_type, crt, count, unitag)
        total += count
        correct += crt
        dict_idx += 1
    print('Total accuracy: ' + str(correct) + ' / ' + str(total))
    db.store_total('total', ml_method, feature_type, correct, total, unitag)









