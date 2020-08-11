from src.core.boundary_condition import DirichletBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import NavierStokesEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4, ExplicitMidpointMethod
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator
from src.core.parareal import PararealOperator
from src.utils.time import time_with_args

diff_eq = NavierStokesEquation(2, 5000.)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.2, .2))
bcs = (
    (DirichletBoundaryCondition(lambda x: (1., .1)),
     DirichletBoundaryCondition(lambda x: (.0, .0))),
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(bvp, lambda x: (.0, .0))
ivp = InitialValueProblem(bvp, (0., 20.), ic)

f = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g = FDMOperator(
    ExplicitMidpointMethod(), ThreePointCentralFiniteDifferenceMethod(), .1)
p = PararealOperator(f, g, 1.)

f_solution_name = 'navier_stokes_fine'
g_solution_name = 'navier_stokes_coarse'
p_solution_name = 'navier_stokes_parareal'

time_with_args(function_name=f_solution_name)(f.solve)(ivp) \
    .plot(f_solution_name, n_images=10)
time_with_args(function_name=g_solution_name)(g.solve)(ivp) \
    .plot(g_solution_name, n_images=10)
time_with_args(function_name=p_solution_name)(p.solve)(ivp) \
    .plot(p_solution_name, n_images=10)
