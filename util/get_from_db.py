import pymysql

# 连接配置信息
config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '',
    'db': 'classdata',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}


def get_summary_db(unitag):
    # 创建连接
    conn = pymysql.connect(**config)
    cur = conn.cursor()
    # 执行sql语句
    try:
        # 执行sql语句，进行查询
        sql = 'SELECT * FROM summary where unitag=\"%s\"'
        cur.execute(sql,unitag)
        # 获取查询结果
        result = cur.fetchall()
        return result
    finally:
        cur.close()
        conn.close()

def get_result_db(unitag):
    # 创建连接
    conn = pymysql.connect(**config)
    cur = conn.cursor()
    # 执行sql语句
    try:
        # 执行sql语句，进行查询
        sql = 'SELECT * FROM result where unitag=\"%s\"'
        cur.execute(sql,unitag)
        # 获取查询结果
        result = cur.fetchall()
        return result
    finally:
        cur.close()
        conn.close()
