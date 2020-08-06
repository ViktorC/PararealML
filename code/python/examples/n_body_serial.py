from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import NBodyGravitationalEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator
from src.utils.time import time

diff_eq = NBodyGravitationalEquation(3, [5e10, 5e12, 5e10])
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(
    bvp,
    lambda _: (-10., 0., 5.) + (0., 0., 0.) + (10., 0., -5.) +
              (0., .25, 0.) + (0., 5., 0.) + (0., -.25, .0))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

solver = ODEOperator('DOP853', 1e-4)
solution = time(solver.solve)(ivp)
solution.plot('fine', n_images=20, smallest_marker_size=20)
