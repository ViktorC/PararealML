import numpy as np

from src.core.constrained_problem import ConstrainedProblem
from src.core.differential_equation import NBodyGravitationalEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator
from src.core.parareal import PararealOperator
from src.utils.time import time_with_args

n_planets = 10
masses = np.random.randint(5e10, 5e12, n_planets)
initial_positions = 40 * np.random.rand(n_planets * 3) - 20.
initial_velocities = 5 * np.random.rand(n_planets * 3)

diff_eq = NBodyGravitationalEquation(3, masses)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: np.append(initial_positions, [initial_velocities]))
ivp = InitialValueProblem(cp, (0., 5.), ic)

f = ODEOperator('RK45', 1e-6)
g = ODEOperator('RK45', 1e-2)
p = PararealOperator(f, g, .5)

f_solution_name = 'n_body_fine'
g_solution_name = 'n_body_coarse'
p_solution_name = 'n_body_parareal'

time_with_args(function_name=f_solution_name)(f.solve)(ivp) \
    .plot(f_solution_name, only_first_process=True)
time_with_args(function_name=g_solution_name)(g.solve)(ivp) \
    .plot(g_solution_name, only_first_process=True)
time_with_args(function_name=p_solution_name)(p.solve)(ivp) \
    .plot(p_solution_name, only_first_process=True)
