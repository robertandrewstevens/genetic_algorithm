# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


def f(x):
    return float(np.sum(x))


var_bound = np.array([[0.5, 1.5], [1, 100], [0, 1]])

var_type = np.array(['real', 'int', 'int'])

model = ga(function=f, dimension=3, variable_type_mixed=var_type, variable_boundaries=var_bound)

model.run()
