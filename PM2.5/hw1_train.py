# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import math
import csv

CSV1 = sys.argv[1]
data = pd.read_csv(CSV1, encoding = 'big5')
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')

for i in range(len(data)):
    if data['測項'][i] == 'NO2':
        pass
    elif data['測項'][i]=='NOx':
        pass
    elif data['測項'][i]=='O3':
        pass
    elif data['測項'][i]=='SO2':
        pass
    elif data['測項'][i]=='PM10':
        pass
    elif data['測項'][i]=='THC':
        pass
    elif data['測項'][i]=='PM2.5':
        pass
    else:
        data = data.drop(index=[i,i])

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
print(raw_data.shape)

month_data = {}
for month in range(12):
    sample = np.empty([7, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[7 * (20 * month + day) : 7 * (20 * month + day + 1), :]
    month_data[month] = sample #12個18*480的數據


x = np.empty([12 * 471, 7 * 9], dtype = float) #行為時間(小時)、列為18個值9個小時測出的值
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][4, day * 24 + hour + 9] #value

mean_x = np.mean(x, axis = 0) #18 * 9 
print(mean_x)
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# x


x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.2): , :]
y_validation = y[math.floor(len(y) * 0.2): , :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

dim = 7 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

#adagrad
# learning_rate = 100
# iter_time = 1000
# adagrad = np.zeros([dim, 1])
# eps = 0.0000000001
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
#     if(t%100==0):
#         print(str(t) + ":" + str(loss))
#     gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
#     adagrad += gradient ** 2
#     w = w - learning_rate * gradient / np.sqrt(adagrad + eps)


# adam
beta1 = 0.9
beta2 = 0.999
learning_rate = 200
eps = 0.0000000001
iter_time = 1000
m_t = 0
v_t = 0
for t in range(iter_time):
    gradient = 2*np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    m_t = beta1 * m_t + (1-beta1) * gradient
    v_t = beta2 * v_t + (1-beta2) * gradient * gradient
    m_head = m_t / (1- np.power(beta1, t+1))
    v_head = v_t / (1- np.power(beta2, t+1))
    w = w - learning_rate* m_head /(np.sqrt(v_head)+eps)

print(w)
np.save('weight.npy', w)
np.save('mean.npy', mean_x)
np.save('std.npy', std_x)
# w

CSV2 = sys.argv[2]
testdata = pd.read_csv(CSV2, header = None, encoding = 'big5')
for i in range(len(testdata)):
    if testdata[1][i] == 'NO2':
        pass
    elif testdata[1][i]=='NOx':
        pass
    elif testdata[1][i]=='O3':
        pass
    elif testdata[1][i]=='SO2':
        pass
    elif testdata[1][i]=='PM10':
        pass
    elif testdata[1][i]=='THC':
        pass
    elif testdata[1][i]=='PM2.5':
        pass
    else:
        testdata = testdata.drop(index=[i,i])

test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

test_x = np.empty([240, 7*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[7 * i: 7* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# test_x

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i][0] < 0:
            ans_y[i][0] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)