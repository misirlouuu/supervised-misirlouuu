import numpy as np
from L1L2Regularization import L1L2Regularization


class LinearRegression:
    def __init__(self, regularization: L1L2Regularization, learning_rate = 0.1, max_iter = 10000):
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n = self.m = self.weights = None

    def gradient_decent(self, x: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.max_iter):
            predictions = np.dot(x, self.weights)
            dw = (1 / self.m) * np.dot(x.transpose(), (predictions - y)) + self.regularization.derivation(self.weights)
            self.weights -= self.learning_rate * dw

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.weights = np.zeros((self.n, 1))
        self.gradient_decent(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights)