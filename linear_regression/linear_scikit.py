import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


data = loadtxt('ex1data2.txt', delimiter=',', dtype=np.float32)

X = data[:, : -1]
y = data[:, -1 :]


def plot_before_feature_normalization(X):
    """
    画出特征缩放前的图
    :param X:
    :return:
    """
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='red')
    plt.show()

plot_before_feature_normalization(X)


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

r_scaler = RobustScaler()
r_scaler.fit(X)
X = r_scaler.transform(X)


def plot_after_feature_normalization(X):
    """
    画出特征缩放后的图
    :param X:
    :return:
    """
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    return X

plot_after_feature_normalization(X)

x_test = scaler.transform(np.array([[1650, 3]], dtype=np.float32))

model = linear_model.LinearRegression()
model.fit(X, y)

result = model.predict(x_test)
print(model.coef_)  # Coefficient of the features 决策函数中的特征系数
print(model.intercept_)  # 又名bias偏置,若设置为False，则为0
print(result[0][0])         # 预测结果
