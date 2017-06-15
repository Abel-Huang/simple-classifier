from main import feature_program as fp
from main import classify_program as cp
from util import data_show as ds
from util import get_from_db as gfd
from util import parse_result as pr
from util import create_unique as cu

def feature_init():
    fp.init_feature_set('brisk')
    fp.learn_vocabulary('brisk')
    fp.init_feature_set('sift')
    fp.learn_vocabulary('sift')
    fp.init_feature_set('surf')
    fp.learn_vocabulary('surf')
    fp.init_feature_set('orb')
    fp.learn_vocabulary('orb')

def classify_model():
    cp.train_classifier('sift')
    cp.train_classifier('surf')
    cp.train_classifier('orb')
    cp.train_classifier('brisk')

def classify_result():
    unitag = cu.create_unique()
    cp.classify('sift', 'svc', unitag)
    cp.classify('surf', 'svc', unitag)
    cp.classify('orb', 'svc', unitag)
    cp.classify('brisk', 'svc', unitag)

    cp.classify('sift', 'rbf', unitag)
    cp.classify('surf', 'rbf', unitag)
    cp.classify('orb', 'rbf', unitag)
    cp.classify('brisk', 'rbf', unitag)

    cp.classify('sift', 'poly', unitag)
    cp.classify('surf', 'poly', unitag)
    cp.classify('orb', 'poly', unitag)
    cp.classify('brisk', 'poly', unitag)

    cp.classify('sift', 'lin', unitag)
    cp.classify('surf', 'lin', unitag)
    cp.classify('orb', 'lin', unitag)
    cp.classify('brisk', 'lin', unitag)
#  生成可视化图形
def visualization_summary(unitag):
    # 查询数据库中的某次实验的总体数据
    data_result=gfd.get_summary_db(unitag)
    sift_list, surf_list, orb_list, brisk_list, name, llabel=pr.parse_summary(data_result)
    ds.show_summary_data(sift_list, surf_list, orb_list, brisk_list, llabel, name)

def visualization_feature(unitag):
    # 查询数据库中的某次实验的特征数据
    data_result=gfd.get_summary_db(unitag)
    sift_list, surf_list, orb_list, brisk_list, name, llabel=pr.parse_fea_result(data_result)
    ds.show_result_data(sift_list, surf_list, orb_list, brisk_list, llabel, name)
    print(sift_list)
    print(surf_list)
    print(orb_list)
    print(brisk_list)
    print(name)
    print(llabel)

def visualization_kernal(unitag):
    # 查询数据库中的某次实验的核函数数据
    data_result=gfd.get_summary_db(unitag)
    sift_list, surf_list, orb_list, brisk_list, name, llabel=pr.parse_ml_result(data_result)
    ds.show_result_data(sift_list, surf_list, orb_list, brisk_list, llabel, name)
    print(sift_list)
    print(surf_list)
    print(orb_list)
    print(brisk_list)
    print(name)
    print(llabel)

if __name__ == '__main__':
    # feature_init()
    # classify_model()
    # classify_result()
    visualization_summary(1496035754193)
    #visualization_kernal(1496035754193)
    # visualization_feature(1496035754193)


