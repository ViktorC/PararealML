import numpy as np

from pararealml import *
from pararealml.core.operators.ode import *
from pararealml.core.operators.parareal import *

diff_eq = LorenzEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
ivp = InitialValueProblem(cp,  (0., 40.), ic)

f = ODEOperator('RK45', 1e-6)
g = ODEOperator('RK45', 1e-5)
p = PararealOperator(f, g, .5)

solution = p.solve(ivp)
solution.plot('lorenz', only_first_process=True)
