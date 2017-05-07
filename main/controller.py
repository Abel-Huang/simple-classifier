from main import feature_program as fp
from main import classify_program as cp
from util import data_show as ds
from util import get_from_db as gfd
from util import parse_result as pr

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
    cp.train_classifier('brisk')
    cp.train_classifier('sift')
    cp.train_classifier('surf')
    cp.train_classifier('orb')

#  生成可视化图形
def visualization(unitag):
    # 查询数据库中的某次实验的总体数据
    data_result=gfd.get_summary_db(unitag)
    print(data_result)
    svc_list, rbf_list, poly_list, liner_list, name, llabel=pr.parse_ml_result(data_result)
    ds.show_result_data(svc_list, svc_list, svc_list, svc_list, name, llabel)


if __name__ == '__main__':
    classify_model()
    # unitag = '1493535328866'
    # visualization(unitag)

