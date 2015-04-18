
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.io as sio


# 读入数据
################################################################################################
print "输入样本文件名（需放在程序目录下）"
filename = 'mnist_train.mat' # raw_input()
sample = sio.loadmat(filename)
sample = sample["mnist_train"]
sample /= 256.0       # 特征向量归一化

print "输入标签文件名（需放在程序目录下）"
filename = 'mnist_train_labels.mat' # raw_input()
label = sio.loadmat(filename)
label = label["mnist_train_labels"]

##################################################################################################


# 神经网络配置
##################################################################################################
samp_num = len(sample)      # 样本总数
inp_num = len(sample[0])    # 输入层节点数
out_num = 10                # 输出节点数
hid_num = 76  # 隐层节点数(经验公式)
w1 = 0.2*np.random.random((inp_num, hid_num))- 0.1   # 初始化输入层权矩阵
w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐层权矩阵
hid_offset = np.zeros(hid_num)     # 隐层偏置向量
out_offset = np.zeros(out_num)     # 输出层偏置向量
inp_lrate = 0.9             # 输入层权值学习率
hid_lrate = 0.9             # 隐层学权值习率
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

def get_err(y, l):
    t_label = np.zeros(len(y))
    t_label[l] = 1
    return 0.5*np.dot((y-t_label), (y-t_label))


###################################################################################################

# 训练
###################################################################################################

for count in range(0, samp_num):
    print count
    #前向过程
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值

    #后向过程
    e = label[count] - out_act                          # 输出值与真值间的误差
    out_delta = e * out_act                             # 输出层delta计算
    for i in range(0, out_num):
        w2[:,i] -= hid_lrate * out_delta[i] * hid_act[i]  # 更新隐层到输出层权向量
    hid_delta = np.dot(w2, out_delta)                   # 隐层delta计算
    out_offset -= hid_lrate * out_delta                             # 输出层偏置更新
    hid_offset -= inp_lrate * hid_delta
    for i in range(0, hid_num):
        w1[:,] -= inp_lrate * hid_delta[i] * sample[count][i]     # 更新输出层到隐层的权向量
###################################################################################################

# 测试网络
###################################################################################################
filename = 'mnist_test.mat'
test = sio.loadmat(filename)
test_s = test["mnist_test"]

filename = 'mnist_test_labels.mat'
testlabel = sio.loadmat(filename)
test_l = testlabel["mnist_test_labels"]
error = 0
for count in range(len(test_s)):
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == test_l[count]:
        error += 1
print float(error)/len(test_s)
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








