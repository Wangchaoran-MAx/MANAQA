import numpy as np
from pandas import DataFrame
from sklearn import linear_model
train_jitter='/hy-tmp/final_test/try_train_jitter.txt'
train_three='/hy-tmp/final_test/try_train_three.txt'
train_three='/hy-tmp/final_test/try_train_score.txt'
val_jitter='/hy-tmp/final_test/try_val_jitter.txt'
val_three='/hy-tmp/final_test/try_val_three.txt'
val_three='/hy-tmp/final_test/try_val_score.txt'
name_data, quat_data, pos_data, NPSS_data, jitter_data = [], [], [], [], []
with open(train_three, 'r') as listFile:
    for line in listFile:
            # 每一行字符串进行切片，默认为空格
        name, quat, pos, NPSS = line.split()
            # [start:end:step]
        quat = float(quat)
        pos = float(pos)
        NPSS = float(NPSS)
        name_data.append(name)
        quat_data.append(quat)
        pos_data.append(pos)
        NPSS_data.append(NPSS)
    #print(np.shape(NPSS_data))
jitter1, name1 = [], []
with open(train_jitter, 'r') as listFile:
    for line in listFile:
            # 每一行字符串进行切片，默认为空格
        name, jitter = line.split()
            # [start:end:step]
        jitter = float(jitter)
        name1.append(name)
        jitter1.append(jitter)
