import math
import numpy as np


class CostFunction:

    def __init__(self, name, fn, dFn):
        self.name = name
        self.fn = fn
        self.dFn = dFn


CostFunction.quadratic = CostFunction(
    name="quadratic",
    fn=lambda actual, expected:
        (actual - expected)**2,
    dFn=lambda actual, expected:
        2 * (actual - expected)
)

CostFunction.half_quadratic = CostFunction(
    name="half_quadratic",
    fn=lambda actual, expected:
        0.5 * (actual - expected)**2,
    dFn=lambda actual, expected:
        actual - expected
)
