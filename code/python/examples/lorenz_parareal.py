from src.core.constrained_problem import ConstrainedProblem
from src.core.differential_equation import LorenzEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator
from src.core.parareal import PararealOperator

diff_eq = LorenzEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: (1., 1., 1.))
ivp = InitialValueProblem(cp,  (0., 40.), ic)

f = ODEOperator('RK45', 1e-6)
g = ODEOperator('RK45', 1e-5)
p = PararealOperator(f, g, .5)

solution = p.solve(ivp)
solution.plot('lorenz', only_first_process=True)
