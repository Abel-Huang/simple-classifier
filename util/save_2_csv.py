import csv
# 将最后的统计结果保存到文件中
csvFile=open("../data/output/hehe.csv","w+",encoding='utf-8')

def save_file():
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('classify', "number plus 2", "number times 2"))
        for i in range(10):
            writer.writerow((i, i + 2, i * 2))
    finally:
        csvFile.close()

