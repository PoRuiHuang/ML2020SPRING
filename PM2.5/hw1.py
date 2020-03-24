# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import math
import csv


CSV1 = sys.argv[1]
testdata = pd.read_csv(CSV1, header = None, encoding = 'big5')
print(testdata.shape)
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
print(test_data.shape)

mean_x = np.load('mean.npy')
std_x  = np.load('std.npy')

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

CSV2 = sys.argv[2]

with open(CSV2, 'w') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i][0] < 0:
            ans_y[i][0] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)