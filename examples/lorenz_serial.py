from src.core.constrained_problem import ConstrainedProblem
from src.core.differential_equation import LorenzEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator
from src.utils.time import time

diff_eq = LorenzEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: (1., 1., 1.))
ivp = InitialValueProblem(cp,  (0., 40.), ic)

solver = ODEOperator('DOP853', 1e-4)
solution = time(solver.solve)(ivp)
solution.plot('lorenz', n_images=50, legend_location='upper right')
