import numpy as np

from src.core.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator
from src.core.parareal import PararealOperator
from src.utils.time import time_with_args

diff_eq = DiffusionEquation(2)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.25, .25))
bcs = (
    (DirichletBoundaryCondition(lambda x: (1.5,)),
     DirichletBoundaryCondition(lambda x: (1.5,))),
    (NeumannBoundaryCondition(lambda x: (0.,)),
     NeumannBoundaryCondition(lambda x: (0.,)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    bvp,
    ((np.array([5., 5.]), np.array([[2.5, 0.], [0., 2.5]])),),
    (100.,))
ivp = InitialValueProblem(bvp, (0., 20.), ic)

f = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .00025)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .00125)
g_g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0125)
p_g = PararealOperator(g, g_g, .1)

p = PararealOperator(f, g, .1)
p_p = PararealOperator(f, p_g, .1)

time_with_args(function_name='original_f_op')(f.solve)(ivp)
time_with_args(function_name='original_g_op')(g.solve)(ivp)
time_with_args(function_name='g_of_parareal_g_op')(g_g.solve)(ivp)
time_with_args(function_name='parareal_g_op')(p_g.solve)(ivp)
time_with_args(function_name='original_parareal_op')(p.solve)(ivp)
time_with_args(function_name='parareal_with_parareal_g_op')(p_p.solve)(ivp)
