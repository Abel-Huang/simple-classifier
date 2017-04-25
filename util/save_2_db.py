import pymysql
# 将每一个测试过的图片信息保存到数据库中
def store_single(picname, classify, mlmethod, feamethod, correct, unitag): ###  strore to database classdata
    try:
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd='', db='classdata', port=3306, charset='utf8')
        cur = conn.cursor()
        cur.execute('INSERT INTO result(picname, classify, mlmethod, feamethod, correct, unitag) VALUES(%s, %s, %s, %s, %s, %s)',
                    (picname, classify, mlmethod, feamethod, correct, unitag));
        cur.connection.commit()
        print('success')
    finally:                          ### close db
        cur.close()
        conn.close()

# 将运行的结果存储到数据库中, 是一个总结信息
def store_total(classify, mlmethod, feamethod, correct, total, unitag):
    try:
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd='', db='classdata', port=3306, charset='utf8')
        cur = conn.cursor()
        cur.execute('INSERT INTO summary(classify, mlmethod, feamethod, correct, total, unitag) VALUES(%s, %s, %s, %s, %s, %s)',
                    (classify, mlmethod, feamethod, correct, total, unitag));
        cur.connection.commit()
        print('success')
    finally:                          ### close db
        cur.close()
        conn.close()


