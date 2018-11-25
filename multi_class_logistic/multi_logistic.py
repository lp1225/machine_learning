import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy import optimize
import random


data = spio.loadmat('ex3data1.mat')
# print(data)
X = data['X']
y = data['y']


def display_data(imageData):
    """
    显示100个数
    初始化一个二维数组，将每行的数据调整成图像矩阵，放进二维数组
    :param imageData:
    :return:
    """
    sum = 0
    pad = 1
    display_array = np.ones((pad+10*(20+pad), pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad + i * (20 + pad):pad + i * (20 + pad) + 20,
                          pad + j * (20 + pad):pad + j * (20 + pad) + 20] = (
                          imageData[sum, :].reshape(20, 20, order="F"))  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.show()


# 随机选取100个数字
rand_indices = random.choices(range(X.shape[0]), k=100)
display_data(X[rand_indices, :])  # 显示100个数字


def sigmoid(z):
    """
    逻辑回归的假设函数
    :param z:
    :return:
    """
    s = 1.0/(1.0+np.exp(-z))
    return s


def regularization_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_coe):
    """
    加入正则化的cost function
    :param X:
    :param y:
    :param theta:
    :param lambda_coe:
    :return:
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    # print('H', h)
    temp = theta.copy()  # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0
    temp[0] = 0
    # 该J已经sum
    J = (np.dot(np.transpose(-y), np.log(h)) - np.dot(np.transpose(1-y), np.log(1-h)))/m \
        + lambda_coe/(2*m) * np.dot(temp.T, temp)

    return J


def regularization_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_coe):
    """
    加入正则化的梯度函数
    :param X:
    :param y:
    :param theta:
    :param lambda_coe:
    :return:
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    temp = theta.copy()
    temp[0] = 0

    grad = np.dot(X.T, (h-y))/m + lambda_coe/m*temp

    return grad


# 因为y，即label是从1到10的，所以下面的alltheta,class_y一定要小心，加一列
def oneVsAll(X, y, num_labels, lambda_coe):
    m, n = X.shape
    alltheta = np.zeros((n+1, num_labels+1))  # n+1行， 11列,从1到10每一列代表一个数字的参数
    # add intercept item
    X = np.hstack((np.ones((m, 1)), X))  # 5000x401
    initial_theta = np.zeros((n+1, 1))  # n+1 x 1

    for one_class in range(1,11,1):

        # reshpe,作用在于将y展开成1x5000的向量
        temp_y = np.int32(y==one_class).reshape((-1,))  # 1x5000
        # 注意：asarray方法将numpy数组转换为普通数组，非常重要，因为用numpy数组会出错，
        temp_y = np.asarray(temp_y)
        # 调用梯度下降的优化方法
        result = optimize.fmin_bfgs(regularization_cost, initial_theta, fprime=regularization_gradient, args=(X, temp_y, lambda_coe))

        alltheta[:, one_class] = result.reshape(1, -1)

    alltheta = alltheta.T
    return alltheta  # 第一行没有使用,11x(n+1) ,每一行代表对一个数字的预测的参数


all_res = oneVsAll(X, y, 10, 0.01)


def predictOneVsAll(all_theta, X):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))  # 5000x401
    p = np.zeros((m, 1))

    k = np.dot(X, all_theta.T)  # 5000x11,第一列是没有用的
    k = k[:, 1:]  # 5000x10
    row_max_index = np.argmax(k, axis=1)
    row_max_index = row_max_index + 1  # 因为是从第一列开始计算的，预测值是1到10
    return row_max_index


pre = predictOneVsAll(all_res, X)  # (5000,)注意这个尺寸是不能和y（5000x1）比较的，故下面转换

precision = np.mean(np.float64(pre.reshape(-1, 1) == y) )
print(precision)

