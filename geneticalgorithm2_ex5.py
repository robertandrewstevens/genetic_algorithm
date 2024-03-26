# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(x):
    return int(np.sum(x))


var_bound = np.array([[0, 10]]*3)

model = ga(function=f, dimension=3, variable_type='int', variable_boundaries=var_bound)

model.run()