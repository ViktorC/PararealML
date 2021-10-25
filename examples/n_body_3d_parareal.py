import numpy as np

from pararealml import *
from pararealml.core.operators.ode import *
from pararealml.core.operators.parareal import *
from pararealml.utils.time import mpi_time

n_planets = 10
masses = np.random.randint(5e4, 5e8, n_planets)
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

mpi_time(f_solution_name)(f.solve)(ivp)[0].plot(
    f_solution_name, only_first_process=True)
mpi_time(g_solution_name)(g.solve)(ivp)[0].plot(
    g_solution_name, only_first_process=True)
mpi_time(p_solution_name)(p.solve)(ivp)[0].plot(
    p_solution_name, only_first_process=True)
