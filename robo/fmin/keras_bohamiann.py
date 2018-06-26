import logging
import numpy as np
import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), "../../../../pysgmcmc_keras/"))

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.optimizers.sghmc2 import SGHMC
from pysgmcmc.optimizers.sghmchd_new import SGHMCHD

from robo.maximizers.direct import Direct
from robo.maximizers.cmaes import CMAES
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.maximizers.random_sampling import RandomSampling
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB


logger = logging.getLogger(__name__)


def bohamiann(objective_function, lower, upper,
              sampling_method=SGHMC, learning_rate=1e-2, mdecay=0.05,
              burn_in_steps=3000, num_steps=50000,
              keep_every=100,
              n_nets=100,
              batch_size=20,
              num_iterations=30, maximizer="random",
              hypergradients_for=("lr", "mdecay", "noise"),
              acquisition_func="log_ei", n_init=3, output_path=None, rng=None):
    """
    Bohamiann uses Bayesian neural networks to model the objective function [1] inside Bayesian optimization.
    Bayesian neural networks usually scale better with the number of function evaluations and the number of dimensions
    than Gaussian processes.

    [1] Bayesian optimization with robust Bayesian neural networks
        J. T. Springenberg and A. Klein and S. Falkner and F. Hutter
        Advances in Neural Information Processing Systems 29

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy array (D,) as input and returns
        the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    num_iterations: int
        The number of iterations (initial design + BO)
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    maximizer: {"direct", "cmaes", "random", "scipy"}
        The optimizer for the acquisition function. NOTE: "cmaes" only works in D > 1 dimensions
    n_init: int
        Number of points for the initial design. Make sure that it is <= num_iterations.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: numpy.random.RandomState
        Random number generator

    Returns
    -------
        dict with all results
    """
    assert upper.shape[0] == lower.shape[0]
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    if sampling_method is SGHMCHD:
        model = BayesianNeuralNetwork(
            hypergradients_for=hypergradients_for,
            optimizer=sampling_method,
            learning_rate=learning_rate,
            mdecay=mdecay,
            burn_in_steps=burn_in_steps,
            num_steps=num_steps,
            keep_every=keep_every,
            n_nets=n_nets,
            batch_size=batch_size,
        )
    else:
        model = BayesianNeuralNetwork(
            optimizer=sampling_method,
            learning_rate=learning_rate,
            mdecay=mdecay,
            burn_in_steps=burn_in_steps,
            num_steps=num_steps,
            keep_every=keep_every,
            n_nets=n_nets,
            batch_size=batch_size,
        )

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)

    else:
        print("ERROR: %s is not a valid acquisition function!" % acquisition_func)
        return

    if maximizer == "cmaes":
        max_func = CMAES(a, lower, upper, verbose=True, rng=rng)
    elif maximizer == "direct":
        max_func = Direct(a, lower, upper, verbose=True)
    elif maximizer == "random":
        max_func = RandomSampling(a, lower, upper, rng=rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(a, lower, upper, rng=rng)

    bo = BayesianOptimization(objective_function, lower, upper, a, model, max_func,
                              initial_points=n_init, output_path=output_path, rng=rng)

    x_best, f_min = bo.run(num_iterations)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = [x.tolist() for x in bo.X]
    results["y"] = [y for y in bo.y]
    return results
