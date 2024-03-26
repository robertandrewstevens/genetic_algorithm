# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


def f(x):
    pen = 0
    if x[0] + x[1] < 2:
        pen = 500 + 1000*(2 - x[0] - x[1])
    return float(np.sum(x) + pen)


var_bound = np.array([[0, 10]]*3)

model = ga(function=f, dimension=3, variable_type='real', variable_boundaries=var_bound)

model.run()

