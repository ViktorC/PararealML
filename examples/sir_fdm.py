import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = SIREquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.array([999., 1., 0.]))
ivp = InitialValueProblem(cp,  (0., 160.), ic)

solver = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralDifferenceMethod(),
    1e-4
)
solution = solver.solve(ivp)

for plot in solution.generate_plots(legend_location='center left'):
    plot.show().close()
