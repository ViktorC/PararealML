import numpy as np

from pararealml import *
from pararealml.operators.ode import *

diff_eq = LorenzEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
ivp = InitialValueProblem(cp,  (0., 40.), ic)

solver = ODEOperator('DOP853', 1e-4)
solution = solver.solve(ivp)
solution.plot('lorenz', n_images=50, legend_location='upper right')
