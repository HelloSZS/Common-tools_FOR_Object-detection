
import os
import random

train_precent=0.8
#base_root = r"C:\Users\29533\Desktop\szs_xc_0406-0408\rain_day_aug"
base_root = os.path.dirname(os.path.abspath(__file__))
print(base_root)


xml=  base_root + "/Annotations/"#Annotations文件夹的路径

total_xml=os.listdir(xml)

num=len(total_xml)
tr=int(num*train_precent)
train=range(0,tr)

train_txt_path = base_root + "/train.txt"
val_txt_path = base_root + "/val.txt"

ftrain=open(train_txt_path,"w")#写你的要存储的train.txt的路径
ftest=open(val_txt_path,"w")#写你的val.txt的路径格式同上

for i in range(num):
    name=total_xml[i][:-4]+"\n"
    if i in train:
        ftrain.write(name)
    else:
        ftest.write(name)

ftrain.close()
ftest.close()