import numpy as np
from mpi4py import MPI

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.parareal import *
from pararealml.utils.time import mpi_time

n_planets = 10
masses = np.random.randint(5e4, 5e8, n_planets)
initial_positions = 40 * np.random.rand(n_planets * 3) - 20.
initial_velocities = 5 * np.random.rand(n_planets * 3)

diff_eq = NBodyGravitationalEquation(3, masses)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: np.append(initial_positions, [initial_velocities])
)
ivp = InitialValueProblem(cp, (0., 5.), ic)

f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 1e-3)
g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 1e-2)
p = PararealOperator(f, g, .5)

f_solution_name = 'n_body_fine'
g_solution_name = 'n_body_coarse'
p_solution_name = 'n_body_parareal'

f_solution, _ = mpi_time(f_solution_name)(f.solve)(ivp)
g_solution, _ = mpi_time(g_solution_name)(g.solve)(ivp)
p_solution, _ = mpi_time(p_solution_name)(p.solve)(ivp)

if MPI.COMM_WORLD.rank == 0:
    for i, plot in enumerate(f_solution.generate_plots()):
        plot.save(f'{f_solution_name}_{i}').close()
    for i, plot in enumerate(g_solution.generate_plots()):
        plot.save(f'{g_solution_name}_{i}').close()
    for i, plot in enumerate(p_solution.generate_plots()):
        plot.save(f'{p_solution_name}_{i}').close()
