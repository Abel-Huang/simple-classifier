import datetime
# 这个文件用来处理数据库返回的数据<class 'list'> list的元素是dict
test_list =[{'mlmethod': 'svc', 'total': 52, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 12, 53), 'unitag': '1493133167337', 'id': 1, 'correct': 49, 'classify': 'glass'},
{'mlmethod': 'svc', 'total': 119, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 3), 'unitag': '1493133167337', 'id': 2, 'correct': 105, 'classify': 'car'},
{'mlmethod': 'svc', 'total': 44, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 7), 'unitag': '1493133167337', 'id': 3, 'correct': 39, 'classify': 'gun'},
{'mlmethod': 'svc', 'total': 63, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 13), 'unitag': '1493133167337', 'id': 4, 'correct': 56, 'classify': 'flowers'},
{'mlmethod': 'svc', 'total': 131, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 25), 'unitag': '1493133167337', 'id': 5, 'correct': 123, 'classify': 'worldcup'},
{'mlmethod': 'svc', 'total': 78, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 32), 'unitag': '1493133167337', 'id': 6, 'correct': 68, 'classify': 'fruits'},
{'mlmethod': 'svc', 'total': 59, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 37), 'unitag': '1493133167337', 'id': 7, 'correct': 57, 'classify': 'city'},
{'mlmethod': 'svc', 'total': 49, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 41), 'unitag': '1493133167337', 'id': 8, 'correct': 48, 'classify': 'dog'},
{'mlmethod': 'svc', 'total': 54, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 46), 'unitag': '1493133167337', 'id': 9, 'correct': 46, 'classify': 'fireworks'},
{'mlmethod': 'svc', 'total': 24, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 48), 'unitag': '1493133167337', 'id': 10, 'correct': 24, 'classify': 'earth'},
{'mlmethod': 'svc', 'total': 78, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 54), 'unitag': '1493133167337', 'id': 11, 'correct': 73, 'classify': 'sky'},
{'mlmethod': 'svc', 'total': 44, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 13, 59), 'unitag': '1493133167337', 'id': 12, 'correct': 40, 'classify': 'gold'},
{'mlmethod': 'svc', 'total': 102, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 14, 6), 'unitag': '1493133167337', 'id': 13, 'correct': 74, 'classify': 'plane'},
{'mlmethod': 'svc', 'total': 897, 'feamethod': 'sift', 'created': datetime.datetime(2017, 4, 25, 23, 14, 6), 'unitag': '1493133167337', 'id': 14, 'correct': 802, 'classify': 'total'}]



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
                    break
                else:
                    continue
        elif r_dict['mlmethod']==llabel[1]:
            for item in name:
                if item==r_dict['classify']:
                    rbf_list.insert(name.index(item),r_dict['correct'])
                    break
                else:
                    continue
        elif r_dict['mlmethod'] == llabel[2]:
            for item in name:
                if item == r_dict['classify']:
                    poly_list.insert(name.index(item), r_dict['correct'])
                else:
                    continue
        elif r_dict['mlmethod'] == llabel[3]:
            for item in name:
                if item == r_dict['classify']:
                    liner_list.insert(name.index(item), r_dict['correct'])
                else:
                    continue
    return svc_list, rbf_list, poly_list, liner_list, name, llabel

svc_list, rbf_list, poly_list, liner_list, _name, _llabel=parse_ml_result(test_list)
print(svc_list)
print(rbf_list)
print(poly_list)
print(liner_list)

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
