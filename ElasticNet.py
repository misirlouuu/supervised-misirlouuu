import numpy as np

from LinearRegression import LinearRegression
from L1L2Regularization import L1L2Regularization


class ElasticNet(LinearRegression):
    def __init__(self, alpha = 0.1, l1_ratio = 0.5, learning_rate = 0.1, max_iter = 10000):
        self.regularization = L1L2Regularization(alpha, l1_ratio)
        super(ElasticNet, self).__init__(self.regularization, learning_rate, max_iter)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        return super(ElasticNet, self).fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return super(ElasticNet, self).predict(x)