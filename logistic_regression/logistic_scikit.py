from numpy import loadtxt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, : -1]
y = data[:, -1 :]
y = y.ravel()
# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 进行标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

logistic = linear_model.LogisticRegression()
logistic.fit(x_train, y_train)
print('权重的值\n', logistic.coef_)  # 权重
print('预测的值\n', logistic.predict(x_test))
print("预测的准确率:", logistic.score(x_test, y_test))
