import numpy as np
from fipy import LinearLUSolver

from src.core.boundary_condition import NeumannBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator

diff_eq = DiffusionEquation(1, 1.5)
mesh = UniformGrid(((0., 10.),), (.1,))
bcs = (
    (NeumannBoundaryCondition(lambda x: (0.,)),
     NeumannBoundaryCondition(lambda x: (0.,))),
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    bvp,
    ((np.array([5.]), np.array([[2.5]])),),
    (20.,))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

solver = FVMOperator(LinearLUSolver(), .01)
solver.solve(ivp).plot('1d_diffusion')
