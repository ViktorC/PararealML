from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LorenzEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator
from src.utils.plot import plot_ivp_solution
from src.utils.time import time

diff_eq = LorenzEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(
    bvp,
    lambda _: (10., 28., 8. / 3.))
ivp = InitialValueProblem(bvp,  (0., 20.), ic)

solver = ODEOperator('DOP853', 1e-4)
solution = time(solver.solve)(ivp)
plot_ivp_solution(solution, 'lorenz', 50, legend_location='upper right')
