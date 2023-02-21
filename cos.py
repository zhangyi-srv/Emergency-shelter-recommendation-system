import csv
from collections import namedtuple
import numpy as np
import pandas as pd

data = pd.read_csv("H:/safe/避难位置数据1.csv", header=0)

# print(data)
# print(data.iloc[1])

i = 1
j = 1  # 循环计算相似度
for i in range(len(data)):
    for j in range(len(data)):
        t1 = np.array(data.iloc[i])
        t2 = np.array(data.iloc[j])
        j += 1

        # print(t1, t2)
        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos


        # print(t1, t2)
        # print(cos_sim(t1, t2))
        # print('共j=%d条' % j)
        aver = np.mean(cos_sim(t1, t2))
        if 1 > aver > 0.99999999:
            print(9,end=" ")
        elif 0.99999999 > aver > 0.99999998:
            print(8,end=" ")
        elif 0.99999998 > aver > 0.99999997:
            print(7,end=" ")
        elif 0.99999997 > aver > 0.99999996:
            print(6,end=" ")
        elif 0.99999996 > aver > 0.99999995:
            print(5,end=" ")
        elif 0.99999995 > aver > 0.99999994:
            print(4,end=" ")
        elif 0.99999994 > aver > 0.99999993:
            print(3,end=" ")
        elif 0.99999993 > aver > 0.99999992:
            print(2,end=" ")
        elif 0.99999992 > aver > 0:
            print(1,end=" ")
    print('共j=%d条' % j)

else:
    print("结束")


# def cos_sim(a, b):
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     cos = np.dot(a, b) / (a_norm * b_norm)
#     return cos
# print(cos_sim(t1, t2))
