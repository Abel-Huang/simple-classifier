import cv2
#用于简化文件操作, 根据参数直接生成文件路径

# 生成分类模型相关的路径
def generic_ml_filename(feature_type, ml_method):
    if ml_method=='svc':
        ml_path='svc_model.m'
    elif ml_method=='rbf':
        ml_path = 'rbf_svc_model.m'
    elif ml_method=='poly':
        ml_path = 'poly_svc_model.m'
    elif ml_method=='lin':
        ml_path = 'lin_svc_model.m'
    filename='../data/model/'+feature_type+'/'+ml_path
    return filename

# 生成特征值相关的路径
def generic_fea_filename(feature_type):
    filename='../data/temp/'+feature_type
    return filename

# 根据参数生成， 关于设置生成点的个数问题
def init_feature_obj(feature_type):
    if feature_type=='sift':
        sift=cv2.xfeatures2d.SIFT_create()
    elif feature_type=='surf':
        sift=cv2.xfeatures2d.SURF_create()
    elif feature_type=='orb':
        sift=cv2.ORB_create()
    elif feature_type=='brisk':
        sift=cv2.BRISK_create()
    return sift

# 生成矩阵维数
def init_feature_dimension(feature_type):
    if feature_type=='sift':
        dimension = 128
    elif feature_type=='surf':
        dimension = 64
    elif feature_type=='orb':
        dimension = 32
    elif feature_type=='brisk':
        dimension = 64
    return dimension

