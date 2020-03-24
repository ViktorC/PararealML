from sklearn.linear_model import LinearRegression

from src.diff_eq import RabbitPopulationDiffEq
from src.operator import ConventionalOperator, MLOperator
from src.integrator import ExplicitMidpointMethod, RK4
from src.parareal import Parareal


diff_eq = RabbitPopulationDiffEq(10000., 5e-2, 0., 100.)
f = ConventionalOperator(RK4(), .025)
g_trainer = ConventionalOperator(ExplicitMidpointMethod(), .1)
g = MLOperator(LinearRegression(), g_trainer, .5, 20)
solver = Parareal(f, g, 2)
solver.solve(diff_eq)
