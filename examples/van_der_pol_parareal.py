from mpi4py import MPI

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.parareal import *
from pararealml.utils.time import mpi_time

diff_eq = VanDerPolEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, vectorize_ic_function(lambda _: [1., 0.]))
ivp = InitialValueProblem(cp,  (0., 20.), ic)

f = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1e-4
)
g = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1e-3
)
p = PararealOperator(f, g, 1e-3)

mpi_time('fine')(f.solve)(ivp)
mpi_time('coarse')(g.solve)(ivp)
solution = mpi_time('parareal')(p.solve)(ivp)[0]

if MPI.COMM_WORLD.rank == 0:
    for plot in solution.generate_plots():
        plot.show().close()
