import numpy as np

from src.core.boundary_condition import DirichletBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import WaveEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator
from src.utils.time import time

diff_eq = WaveEquation(2)
mesh = UniformGrid(((-5., 5.), (-5., 5.)), (.1, .1))
bcs = (
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0))),
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    bvp,
    ((np.array([0., 2.5]), np.array([[.1, 0.], [0., .1]])),) * 2,
    (3., .0))
ivp = InitialValueProblem(bvp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
solution = time(solver.solve)(ivp)
solution.plot('wave_equation', n_images=50)
