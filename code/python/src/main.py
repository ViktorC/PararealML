from src.diff_eq import RabbitPopulationDiffEq
from src.operator import ConventionalOperator
from src.integrator import ForwardEulerMethod, RK4
from src.parareal import Parareal


diff_eq = RabbitPopulationDiffEq(10000., 5e-2, 0., 1.)
f = ConventionalOperator(RK4(), .025)
g = ConventionalOperator(ForwardEulerMethod(), .1)
solver = Parareal(f, g, 2)
solver.solve(diff_eq)
