import numpy as np


class GradientDescent:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, x, dx):
        return x - self.learning_rate * dx
