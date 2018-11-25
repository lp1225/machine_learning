import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt


data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, : -1]
y = data[:, -1 :]


def plot_data(X, y):
    """
    画散点图
    :param X:
    :param y:
    :return:
    """
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], marker='_')
    plt.show()


plot_data(X, y)


def feature_normalization(X):
    """
    特征缩放
    :param X:
    :return:
    """
    X_norm = X

    column_mean = np.mean(X_norm, axis=0) # 竖着看，按列
    print('mean=', column_mean)
    column_std = np.std(X_norm, axis=0)
    print('std=', column_std)

    X_norm = X_norm - column_mean
    X_norm = X_norm / column_std

    return column_mean, column_std, X_norm


means, stds, X_norm = feature_normalization(X)
plot_data(X_norm, y)

X = X_norm
X = np.hstack((np.ones((len(y), 1)), X))


def sigmoid(z):
    """
    逻辑回归方程
    :param z:
    :return:
    """
    s = 1.0/(1.0+np.exp(-z))
    return s


def gradient_1(theta, X, y):
    """
    求一阶导数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    z = np.dot(X, theta)
    p1 = sigmoid(z)
    g = -np.dot(X.T, (y-p1))

    return g


initial_theta = np.zeros((X.shape[1],1))  # 初始化theta
# grad1 = gradient_1(initial_theta,X, y)
# print(grad1)


def target_func_maxmium_likelihood(theta, X, y):
    """
    X包含的偏置项
    :param theta:
    :param X:
    :param y:
    :return:
    """
    l = np.sum(-y*np.dot(X, theta)+np.log(1+np.exp(np.dot(X, theta))))

    return l


target_func_maxmium_likelihood(initial_theta, X, y)


alpha = 0.001
iterations = 400


def gradient_descent(X, y, theta, alpha, iterations):
    """
    梯度下降
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param iterations:
    :return:
    """
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        grad = gradient_1(theta, X, y)
        theta = theta-alpha*grad
        J_history[i] = target_func_maxmium_likelihood(theta, X, y)

    return theta, J_history


theta, J_history= gradient_descent(X, y, initial_theta, alpha, iterations)


def plot_j(J_history, iterations):
    x = np.arange(1, iterations+1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations vs loss')
    plt.show()


plot_j(J_history, iterations)

