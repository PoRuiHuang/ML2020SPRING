# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

def mean(x):
  return sum(x) / len(x)

def de_mean(x):
  x_bar = mean(x)
  return [x_i - x_bar for x_i in x]
# 輔助計算函式 dot product 、sum_of_squares
def dot(v, w):
  return sum(v_i * w_i for v_i, w_i in zip(v, w))
def sum_of_squares(v):
  return dot(v, v)
# 方差
def variance(x):
  n = len(x)
  deviations = de_mean(x)
  return sum_of_squares(deviations) / (n - 1)
# 標準差
def standard_deviation(x):
  return math.sqrt(variance(x))

def covariance(x, y):
  n = len(x)
  return dot(de_mean(x), de_mean(y)) / (n -1)

def correlation(x, y):
  stdev_x = standard_deviation(x)
  stdev_y = standard_deviation(y)
  if stdev_x > 0 and stdev_y > 0:
    return covariance(x, y) / stdev_x / stdev_y
  else:
    return 0

d1 = sys.argv[1]
d2 = sys.argv[2]
d3 = sys.argv[3]
d4 = sys.argv[4]
d5 = sys.argv[5]
d6 = sys.argv[6]
# CSV1 = sys.argv[1]
# data = pd.read_csv(CSV1, encoding = 'big5')
data = pd.read_csv(d1, encoding = 'big5')
# data.drop(columns=['id', 'migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)
# print(data.columns.values.tolist() )
# print(data["education"])
# print(data.shape)
# data_dum = pd.get_dummies(data)
# pd.DataFrame(data_dum)
# X_train = data_dum.to_numpy(dtype = float)
# print(X_train.shape)
np.random.seed(0)
X_train_fpath =  d3
Y_train_fpath = d4
X_test_fpath = d5
output_fpath = d6

drop_list = []
with open(Y_train_fpath) as f:
    # next(f)
    # Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    # print(Y_train.shape)
    # print(type(Y_train))
    # print(Y_train)


    data2 = pd.read_csv(f)
    # print(data2.shape)
    data2 = data2.iloc[:, 1:]
    # data2.drop(columns=['migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)

    # next(f)
    Y_train = data2.to_numpy(dtype = float)
    Y_train  = Y_train.reshape((len(Y_train),))
# Parse csv files to numpy array
with open(X_train_fpath) as f:
    # next(f)
    # X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    # print(X_train.shape)
    # print(X_train)

    data1 = pd.read_csv(f)
    # print(data1.shape)

    data1 = data1.iloc[:, 1:]
    # data1.drop(columns=['migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)
    # data.drop(columns=['id', 'migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)
    X_train = data1.to_numpy(dtype = float)
    
    for i in range(len(X_train[0])):
        corr = correlation(X_train[:,i],Y_train)
        # print(i,corr,sep=": ")
        if(abs(corr) < 0.01):
            drop_list.append(i)
    
    print(drop_list)
    # print(len(drop_list))
    print(X_train.shape)
    X_train = np.delete(X_train,drop_list,1) 


# with open(Y_train_fpath) as f:
#     # next(f)
#     # Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
#     # print(Y_train.shape)
#     # print(type(Y_train))
#     # print(Y_train)


#     data2 = pd.read_csv(f)
#     # print(data2.shape)
#     data2 = data2.iloc[:, 1:]
#     # data2.drop(columns=['migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)

#     # next(f)
#     Y_train = data2.to_numpy(dtype = float)
#     Y_train  = Y_train.reshape((len(Y_train),))
#     print(Y_train.shape)
with open(X_test_fpath) as f:
    # next(f)
    # X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    # print(X_test.shape)
    # print(X_test)

    data3 = pd.read_csv(f)
    # print(data3.shape)
    data3 = data3.iloc[:, 1:]
    # data3.drop(columns=['migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt','country of birth father','country of birth mother','country of birth self'],inplace=True)
    X_test = data3.to_numpy(dtype = float)
    X_test = np.delete(X_test,drop_list,1) 



def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
        # print(specified_column.shape)
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
        # print(X_mean)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# print(X_train)
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
print(X_mean.shape)
print(X_std.shape)
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

lambda_c = 100

def _cross_entropy_loss(y_pred, Y_label,w):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred)) + 2*lambda_c*w*w/len(w)
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad
# Parse csv files to numpy array
# with open(X_train_fpath) as f:
#     next(f)
#     X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
# with open(Y_train_fpath) as f:
#     next(f)
#     Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
# with open(X_test_fpath) as f:
#     next(f)
#     X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)



# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)  

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])



# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))


# Predict testing labels
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
