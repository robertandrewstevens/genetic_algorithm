# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


def f(x):
    return float(np.sum(x))


var_bound = np.array([[0, 10]]*3)

algorithm_param = {'max_num_iteration': 3000,
                   'population_size': 100,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'mutation_type': 'uniform_by_center',
                   'selection_type': 'roulette',
                   'max_iteration_without_improv': None}

model = ga(
    function=f,
    dimension=3,
    variable_type='real',
    variable_boundaries=var_bound,
    algorithm_parameters=algorithm_param
)

model.run()
