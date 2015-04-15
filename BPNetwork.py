
# -*- coding: utf-8 -*-

import random
import math
import numpy as np


# 读入数据
################################################################################################
print "输入样本文件名（需放在程序目录下）"
filename=raw_input()
try:
    f=open(filename,'r')

except IOError:
    print "Wrong input!"
    exit(1)

# 读取文件里的数据，转为list
def GetData(s):
    k = []
    for i in s:
        if i.isdigit():
            k.append(int(i))
    return k

# sample是样本矩阵，label是对应的标签向量
sample = []
label = []
for line in f:
    temp = GetData(line)        # 读取数据
    label.append(temp.pop())    # 读取标签并弹出
    temp.append(1)              # 增加偏置项1
    sample.append(temp)         # 样本读入到sample中

f.close()

sample = np.array(sample)
label = np.array(label)
##################################################################################################


# 神经网络配置
##################################################################################################
samp_num = len(sample)      # 样本总数
inp_num = len(sample[0])    # 输入层节点数
hid_num = inp_num-1         # 隐层节点数
out_num = 10                # 输出节点数
w1 = 0.2*np.random.random((inp_num, hid_num))-0.1   # 初始化输入层权矩阵
w2 = 0.2*np.random.random((hid_num, out_num))-0.1   # 初始化隐层权矩阵
delta2 = np.zeros(out_num)                 # 隐层到输出层的delta
delta1 = np.zeros(hid_num)                 # 输入层到隐层的delta
inp_lrate = 0.5             # 输入层学习率
hid_lrate = 0.5             # 隐层学习率
err_th = 0.01                # 学习误差门限


###################################################################################################

# 必要函数定义
###################################################################################################
def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def get_err(y, l):
    tem_label = np.zeros(len(y))
    tem_label[l] = 1
    return (0.5)*np.dot((y-tem_label), (y-tem_label))


###################################################################################################

# 训练
###################################################################################################

for count in range(0, samp_num):
    hid_value = np.dot(sample[count], w1)       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2)             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    err = get_err(out_act, label[count])        # 计算误差

    if err <= err_th:
        print "Training finished, OK"
    else:
        for i in range(0, len(delta2)):
            delta2[i] = (label[count] - out_act[count]) * sigmoid(out_value[i]) * (1 - sigmoid(out_value[i]))        # 输出层delta
            w2[:, i] = w2[:, i] - hid_lrate * delta2[i] * hid_act  # 更新隐层到输出层权向量
        for i in range(0, len(delta1)):
            delta1[i] = sigmoid(hid_value[i]) * (1 - sigmoid(hid_value[i])) * np.dot(delta2, w2[i])  # 隐层delta
            w1[:, i] = w1[:, i] - inp_lrate * delta1[i] * sample[i]  # 更新隐层到输出层权向量
###################################################################################################

# 输出网络
###################################################################################################
Network = open("MyNetWork", 'w')
Network.write(str(inp_num))
Network.write('\n')
Network.write(str(hid_num))
Network.write('\n')
Network.write(str(out_num))
Network.write('\n')
for i in w1:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
    Network.write('\n')
Network.write('\n')

for i in w2:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
Network.write('\n')

Network.close()









