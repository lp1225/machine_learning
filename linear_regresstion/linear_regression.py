import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt


data = loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]  # 预测值 (97,)
y = data[:, 1]  # 真实值

plt.scatter(X, y, marker='x')
plt.show()

m = len(y)
X = np.reshape(X, (m, 1))  # 对X进行转置,(97, 1)
y = np.reshape(y, (m, 1))

X = np.hstack((np.ones((m, 1)), X))  # 合并两个同维度的数组, 添加1, 后面需要转置
theta = np.zeros((2, 1))


def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    代价函数
    :param X:
    :param y:
    :param theta:
    :return:
    """
    m = len(y)
    J = 0
    h = np.dot(X, theta)  # 矩阵乘法
    J = 1/(2*m)*sum((h-y)**2)
    return J[0]


# cost_function(X, y, theta)  # 初始化
# cost_function(X, y, np.array([[-1], [2]]))   # 则h(x) = -1 + 2x

iterations = 1500  # 训练次数
alpha = 0.01


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
    J_history = np.zeros((iterations, 1))  # 存放每次的损失值

    for i in range(iterations):
        h = np.dot(X, theta) # 预测值
        k = np.dot(np.transpose(X), (h-y))
        theta = theta - alpha * k / m   # 每次更新theta值
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history, k


theta, J_history, k = gradient_descent(X, y, theta, alpha, iterations)


def plot_j(J_history, iterations):
    """
    画cost function
    :param J_history:
    :param iterations:
    :return:
    """
    x = np.arange(1, iterations+1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations vs loss')
    plt.show()


plot_j(J_history, iterations)


def plot_result(X, y, theta):
    """
    画出回归的图
    """
    plt.scatter(X[:, 1], y)
    plt.plot(X[:, 1], np.asarray(np.dot(X, theta)), color='r')  # 根据theta和x来画图, 这里使用array和asarray是一样的
    plt.show()


plot_result(X, y, theta)



