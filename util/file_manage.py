#用于简化文件操作, 根据参数直接生成文件路径

# 获取分类模型相关的路径
def generic_ml_filename(ml_method, feature_type):
    filename='../data/model/'+feature_type+'/'+ml_method+'_model.m'
    return filename

# 获取特征值相关的路径
def generic_fea_filename(feature_type):
    filename='../data/temp/'+feature_type+'/'
    return filename