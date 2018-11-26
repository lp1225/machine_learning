import numpy as np


class Perceptron:
    """
    a simple neural network
    """
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length)*0.5
        else:
            self.weights = weights

    @staticmethod
    def unit_step_function(x):
        """
        g为sigmoid
        :param x:
        :return:
        """
        if x > 0.5:
            return 1
        return 0

    def __call__(self, in_data):
        """
        一个类实例也可以变成一个可调用对象
        """
        weighted_input = self.weights*in_data
        weighted_sum=weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)


p = Perceptron(2, np.array([0.5, 0.5]))
for x in [np.array([0, 0]), np.array([0, 1]),
          np.array([1, 0]), np.array([1, 1])]:
    y = p(np.array(x))
    print(x, y)