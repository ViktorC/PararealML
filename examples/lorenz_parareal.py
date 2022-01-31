import numpy as np
from mpi4py import MPI

from pararealml import *
from pararealml.operators.ode import *
from pararealml.operators.parareal import *

diff_eq = LorenzEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
ivp = InitialValueProblem(cp,  (0., 40.), ic)

f = ODEOperator('RK45', 1e-6)
g = ODEOperator('RK45', 1e-5)
p = PararealOperator(f, g, .5)

solution = p.solve(ivp)

if MPI.COMM_WORLD.rank == 0:
    for i, plot in enumerate(solution.generate_plots()):
        plot.save(f'lorenz_parareal_{i}').close()

