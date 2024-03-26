# https://pypi.org/project/geneticalgorithm2

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model
from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior
from geneticalgorithm2 import Population_initializer # for creating better start population
from geneticalgorithm2 import np_lru_cache # for cache function (if u want)
from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want
from geneticalgorithm2 import Callbacks # simple callbacks
from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks


def function(x):  # x as numpy array
    return float(np.sum(x**2) + x.mean() + x.min() + x[0]*x[2])  # some float result


var_bound = np.array([[0, 10]]*3)  # 2D numpy array

model = ga(
    function=function,
    dimension=3,
    variable_type='real',
    variable_boundaries=var_bound,
    variable_type_mixed=None,
    function_timeout=10,
    algorithm_parameters={
        'max_num_iteration': None,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'mutation_type': 'uniform_by_center',
        'selection_type': 'roulette',
        'max_iteration_without_improv': None
    }
)

model.run(
    no_plot=False,
    disable_progress_bar=False,
    set_function=None,
    apply_function_to_parents=False,
    start_generation={'variables': None, 'scores': None},
    studEA=False,
    mutation_indexes=None,
    init_creator=None,
    init_oppositors=None,
    duplicates_oppositor=None,
    remove_duplicates_generation_step=None,
    revolution_oppositor=None,
    revolution_after_stagnation_step=None,
    revolution_part=0.3,
    population_initializer=Population_initializer(
        select_best_of=1,
        local_optimization_step='never',
        local_optimizer=None
    ),
    stop_when_reached=None,
    callbacks=[],
    middle_callbacks=[],
    time_limit_secs=None,
    save_last_generation_as=None,
    seed=None
)
