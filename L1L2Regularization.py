import numpy as np


class L1L2Regularization:
    def __init__(self, alpha = 0.1, l1_ratio = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        l1_penalty = self.l1_ratio * self.alpha * np.sum(np.abs(weights))
        l2_penalty = (1 - self.l1_ratio) * self.alpha * 0.5 * np.sum(np.square(weights))
        return l1_penalty + l2_penalty

    def derivation(self, weights: np.ndarray) -> np.ndarray:
        l1_derivation = self.alpha * self.l1_ratio * np.sign(weights)
        l2_derivation = self.alpha * (1 - self.l1_ratio) * weights
        return l1_derivation + l2_derivation