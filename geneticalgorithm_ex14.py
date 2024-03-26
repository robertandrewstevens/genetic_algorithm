# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from OptimizationTestFunctions import Eggholder


dim = 2*15

f = float(Eggholder(dim))

xmin, xmax, ymin, ymax = f.bounds

var_bound = np.array([[xmin, xmax], [ymin, ymax]]*15)

model = ga(
    function=f,
    dimension=dim,
    variable_type='real',
    variable_boundaries=var_bound,
    algorithm_parameters = {
        'max_num_iteration': 300,
        'population_size': 100
    }
)

# first run and save last generation to file
filename = "eggholder_lastgen.npz"
model.run(save_last_generation_as=filename)

# load start generation from file and run again (continue optimization)
model.run(start_generation=filename)
