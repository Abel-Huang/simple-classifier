import os
import re

# 文件管理工具
fnLike=re.compile('[a-z]{1}')

absPath = r'H:\data' #要修改文件所处路径
all_file_list = os.listdir(absPath) #列出指定目录下的所有文件
Oldpart = "a" #要替换的文件名中的部分
Newpart = "b" #新的文件名部分

Oldpostfix =r".txt" #要修改的文件扩展名类型
Newpostfix = r".Grey" #新的文件扩展名类型

#批量修改文件名字
def modify_prefix(oldcontent):
   index=1
   for file_name in all_file_list:
       currentdir =os.path.join(absPath, file_name) #连接指定的路径和文件名or文件夹名字
       if os.path.isdir(currentdir): #如果当前路径是文件夹，则跳过
          continue
       fname = os.path.splitext(file_name)[0] #分解出当前的文件路径名字
       ftype = os.path.splitext(file_name)[1] #分解出当前的文件扩展名
       replname =fname.replace(oldcontent,'%d'%index+'refl')
       index+=index+1
       newname = os.path.join(absPath, replname+ftype) #文件路径与新的文件名字+原来的扩展名
       os.rename(currentdir, newname) #重命名
   print("Modify file name........")


#批量修改文件扩展名
def modify_postfix(oldftype, newftype):
    for file_name in all_file_list:
        currentdir = os.path.join(absPath, file_name)
        if os.path.isdir(currentdir): #跳过文件夹
           continue
        fname = os.path.splitext(file_name)[0]
        ftype = os.path.splitext(file_name)[1]
        if ftype == oldftype:  #找到需要修改的扩展名
           newname = os.path.join(absPath, fname+newftype) #文件路径与原来的文件名字+新的扩展名
           os.rename(currentdir, newname) #重命名
    print("Modify file postfix...... ")
modify_postfix(Oldpostfix, Newpostfix)

modify_prefix(fnLike)
print("finished !")