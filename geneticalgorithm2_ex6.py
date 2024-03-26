# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


def f(x):
    return int(np.sum(x))


model = ga(function=f, dimension=30, variable_type='bool')

model.run()
