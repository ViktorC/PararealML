from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.parareal import *
from pararealml.utils.time import time_with_args

diff_eq = NavierStokesStreamFunctionVorticityEquation(5000.)
mesh = Mesh(((0., 10.), (0., 10.)), (.2, .2))
bcs = (
    (DirichletBoundaryCondition(lambda x, t: (1., .1), is_static=True),
     DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True)),
    (DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True),
     DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda x: (.0, .0))
ivp = InitialValueProblem(cp, (0., 20.), ic)

f = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g = FDMOperator(
    ExplicitMidpointMethod(), ThreePointCentralFiniteDifferenceMethod(), .1)
p = PararealOperator(f, g, 1.)

f_solution_name = 'navier_stokes_fine'
g_solution_name = 'navier_stokes_coarse'
p_solution_name = 'navier_stokes_parareal'

time_with_args(function_name=f_solution_name)(f.solve)(ivp) \
    .plot(f_solution_name, only_first_process=True, n_images=10)
time_with_args(function_name=g_solution_name)(g.solve)(ivp) \
    .plot(g_solution_name, only_first_process=True, n_images=10)
time_with_args(function_name=p_solution_name)(p.solve)(ivp) \
    .plot(p_solution_name, only_first_process=True, n_images=10)
