from util import data_show as ds
from util import get_from_db as gfd
from util import parse_result as pr

#  生成可视化图形
def visualization(unitag):
    # 查询数据库中的某次实验的总体数据
    data_result=gfd.get_summary_db(unitag)
    print(data_result)
    svc_list, rbf_list, poly_list, liner_list, name, llabel=pr.parse_ml_result(data_result)
    print(llabel)
    ds.show_result_data(svc_list, svc_list, svc_list, svc_list, name, llabel)

unitag=1493133167337
visualization(unitag)