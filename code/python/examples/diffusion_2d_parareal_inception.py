import numpy as np
from fipy import LinearLUSolver

from src.core.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator
from src.core.parareal import PararealOperator
from src.utils.time import time_with_args

diff_eq = DiffusionEquation(2)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.5, .5))
bcs = (
    (DirichletBoundaryCondition(lambda x: (1.5,)),
     DirichletBoundaryCondition(lambda x: (1.5,))),
    (NeumannBoundaryCondition(lambda x: (0.,)),
     NeumannBoundaryCondition(lambda x: (0.,)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    bvp,
    ((np.array([5., 5.]), np.array([[1., 0.], [0., 1.]])),),
    (1000.,))
ivp = InitialValueProblem(bvp, (0., 8.), ic)

f = FVMOperator(LinearLUSolver(), .00125)
g = FVMOperator(LinearLUSolver(), .0125)
g_g = FVMOperator(LinearLUSolver(), .5)
p_g = PararealOperator(g, g_g, .0, max_iterations=2)

p = PararealOperator(f, g, .01)
p_p = PararealOperator(f, p_g, .01)
p_g_g = PararealOperator(f, g_g, .01)

time_with_args(function_name='original_f_op')(f.solve)(ivp)
time_with_args(function_name='original_g_op')(g.solve)(ivp)
time_with_args(function_name='g_of_parareal_g_op')(g_g.solve)(ivp)
time_with_args(function_name='parareal_g_op')(p_g.solve)(ivp)
time_with_args(function_name='original_parareal_op')(p.solve)(ivp)
time_with_args(function_name='parareal_with_parareal_g_op')(p_p.solve)(ivp)
time_with_args(function_name='parareal_with_g_of_parareal_g_op')(p_g_g.solve)(
    ivp)
