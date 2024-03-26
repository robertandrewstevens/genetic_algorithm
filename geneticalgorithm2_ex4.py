# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


def f(x):
    return float(np.sum(x))


var_bound = np.array([[0, 10]]*3)

model = ga(function=f, dimension=3, variable_type='real', variable_boundaries=var_bound)
model.run()

# y = [1, 1, 1]
# print(y)
# print(f(y))
# print(type(f(y)))
