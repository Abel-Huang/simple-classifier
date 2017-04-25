import json

# 这个文件用来处理数据库返回的数据<class 'list'> list的元素是dict
json_dict = {'feamethod': "sift", 'mlmethod': "svc", 'classify': "car", 'correct': 1, 'total': 10}

# 这个函数用于解析采用不同核函数的数据
def parse_ml_result(result_list):
    svc_list=[]
    rbf_list=[]
    poly_list=[]
    liner_list=[]

    name = ('car', 'city', 'dog', 'earth', 'fireworks', 'flowers',
            'fruits', 'glass', 'gold', 'gun', 'plane', 'sky', 'worldcup')
    llabel = ('svc', 'rbf_svc', 'poly_svc', 'lin_svc')
    # 这里的每一个元素都是一条数据
    for r_dict in result_list:
        if r_dict['mlmethod']==llabel[0]:
            for item in name:
                if item==r_dict['classify']:
                    svc_list.insert(name.index(item),r_dict['correct'])
        elif r_dict['mlmethod']==llabel[1]:
            for item in name:
                if item==r_dict['classify']:
                    rbf_list.insert(name.index(item),r_dict['correct'])
        elif r_dict['mlmethod'] == llabel[2]:
            for item in name:
                if item == r_dict['classify']:
                    poly_list.insert(name.index(item), r_dict['correct'])
        elif r_dict['mlmethod'] == llabel[3]:
            for item in name:
                if item == r_dict['classify']:
                    liner_list.insert(name.index(item), r_dict['correct'])
    return svc_list, rbf_list, poly_list, liner_list, name, llabel

# 这个函数用于解析采用不同特征提取的数据
def parse_fea_result(result_list):
    svc_list = []
    rbf_list = []
    poly_list = []
    liner_list = []

    name = ('car', 'city', 'dog', 'earth', 'fireworks', 'flowers',
            'fruits', 'glass', 'gold', 'gun', 'plane', 'sky', 'worldcup')
    llabel = ('svc', 'rbf_svc', 'poly_svc', 'lin_svc')
    # 这里的每一个元素都是一条数据
    for r_dict in result_list:
        if r_dict['mlmethod'] == llabel[0]:
            for item in name:
                if item == r_dict['classify']:
                    svc_list.insert(name.index(item), r_dict['correct'])
        elif r_dict['mlmethod'] == llabel[1]:
            for item in name:
                if item == r_dict['classify']:
                    rbf_list.insert(name.index(item), r_dict['correct'])
        elif r_dict['mlmethod'] == llabel[2]:
            for item in name:
                if item == r_dict['classify']:
                    poly_list.insert(name.index(item), r_dict['correct'])
        elif r_dict['mlmethod'] == llabel[3]:
            for item in name:
                if item == r_dict['classify']:
                    liner_list.insert(name.index(item), r_dict['correct'])
    return svc_list, rbf_list, poly_list, liner_list, name, llabel

# parse_result(json_str)
