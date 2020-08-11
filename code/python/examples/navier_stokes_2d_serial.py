from src.core.boundary_condition import DirichletBoundaryCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import NavierStokesEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator
from src.utils.time import time

diff_eq = NavierStokesEquation(2, 5000.)
mesh = UniformGrid(((-2.5, 2.5), (0., 4.)), (.05, .05))
bcs = (
    (DirichletBoundaryCondition(lambda x: (1., .1)),
     DirichletBoundaryCondition(lambda x: (.0, .0))),
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(bvp, lambda x: (.0, .0))
ivp = InitialValueProblem(bvp, (0., 100.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .02)
solution = time(solver.solve)(ivp)
solution.plot('navier_stokes', n_images=50)
