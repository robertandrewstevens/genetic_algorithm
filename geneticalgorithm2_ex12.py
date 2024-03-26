# https://pypi.org/project/geneticalgorithm2
# How to initialize start population? How to continue optimization with new run?
# For this there is `start_generation` parameter in `run()` method.
# It's the dictionary with structure like returned `model.output_dict['last_generation']`.

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(x):
    return float(np.sum(x))


dim = 6
var_bound = np.array([[0, 10]]*dim)

algorithm_param = {'max_num_iteration': 500,
                   'population_size': 100,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'mutation_type': 'uniform_by_center',
                   'selection_type': 'roulette',
                   'max_iteration_without_improv': None}

model = ga(function=f,
           dimension=dim,
           variable_type='real',
           variable_boundaries=var_bound,
           algorithm_parameters=algorithm_param)

# start generation
# use any values valid for a function
samples = np.random.uniform(0, 50, (300, dim))  # 300 is the new size of your generation
# samples = np.random.uniform(0, 50, (1, dim))  # 300 is the new size of your generation
# samples = np.array([0.00590301, 0.00047326, 0.00229967, 0.00248911, 0.00570256, 0.00676421])
# samples = samples.reshape(1, 6)
print(type(samples))
print(samples.shape)
print(samples)

model.run(no_plot=True, start_generation={'variables': samples, 'scores': None})
# it's not necessary to evaluate scores before
# but you can do it if you have evaluated scores and don't wanna repeat calculations
## best value = 0.10426190111045064

print(type(model.output_dict['last_generation']))
print(len(model.output_dict['last_generation']))
print(model.output_dict['last_generation'])

# continue optimization using saved last generation
model.run(no_plot=True, start_generation=model.output_dict['last_generation'])
# model.run(no_plot=True, start_generation=samples)
## best value = 0.06128462776296528
