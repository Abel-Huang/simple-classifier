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


def get_db():
    # 创建连接
    conn = pymysql.connect(**config)
    cur = conn.cursor()
    # 执行sql语句
    try:
        # 执行sql语句，进行查询
        sql = 'SELECT * FROM result'
        cur.execute(sql)
        # 获取查询结果
        result = cur.fetchall()
        print(result)
        print(type(result))
    finally:
        cur.close()
        conn.close()

get_db()
