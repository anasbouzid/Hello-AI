import math
import numpy as np


class ActivationFunction:

    def __init__(self, name, fn, dFn):
        self.name = name
        self.fn = fn
        self.dFn = dFn


def iterativeCalls(func, data):
    if isinstance(data, (list, np.ndarray)):
        return np.array(
            [func(datum) for datum in data]
        )
    return func(data)


def sigmoidFunc(x): return 1 / (1 + math.exp(-x))
def sigmoidDerivativeFunc(x): return sigmoidFunc(x) * (1 - sigmoidFunc(x))

ActivationFunction.sigmoid = ActivationFunction(
    name="sigmoid",
    fn=lambda x: iterativeCalls(sigmoidFunc, x),
    dFn=lambda x: iterativeCalls(sigmoidDerivativeFunc, x)
)


def reLU_func(x): return 0 if x <= 0 else x
def reLU_derivative_func(x): return 0 if x <= 0 else 1

ActivationFunction.reLU = ActivationFunction(
    name="ReLU",
    fn=lambda x: iterativeCalls(reLU_func, x),
    dFn=lambda x: iterativeCalls(reLU_derivative_func, x)
)


def leaky_reLU_func(x): return 0.01 * x if x <= 0 else x
def leaky_reLU_derivative_func(x): return 0.01 if x <= 0 else 1

ActivationFunction.leaky_reLU = ActivationFunction(
    name="leaky_ReLU",
    fn=lambda x: iterativeCalls(leaky_reLU_func, x),
    dFn=lambda x: iterativeCalls(leaky_reLU_derivative_func, x)
)


ActivationFunction.identity = ActivationFunction(
    name="identity",
    fn=lambda x: x,
    dFn=lambda x: 1
)

"""
Numeric stability: avoid computing large exponential values and still get the same results, see:
https://cs231n.github.io/linear-classify/#softmax
"""
def softmax_func(vec):
    vec = vec - np.max(vec)
    return np.exp(vec) / np.sum(np.exp(vec))

"""
https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
"""
def softmax_derivative_func(vec):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = vec.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

ActivationFunction.softmax = ActivationFunction(
    name="softmax",
    fn=lambda x: softmax_func(x),
    dFn=lambda x: softmax_derivative_func(x)
)
