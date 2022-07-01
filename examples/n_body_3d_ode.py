import numpy as np

from pararealml import *
from pararealml.operators.ode import *

# Initial conditions taken from
# https://www.aanda.org/articles/aa/full/2002/08/aa1405/aa1405.right.html

astronomical_unit = 1.496e11
day = 24 * 3600

masses = [1.989e30, 3.301e23, 4.867e24, 6.046e24, 6.417e23]
positions_au = [
    0e0,
    0e0,
    0e0,
    -2.503321047836e-1,
    1.873217481656e-1,
    1.260230112145e-1,
    1.747780055994e-2,
    -6.624210296743e-1,
    -2.991203277122e-1,
    -9.091916173950e-1,
    3.592925969244e-1,
    1.557729610506e-1,
    1.203018828754e0,
    7.270712989688e-1,
    3.009561427569e-1,
]
velocities_au_d = [
    0e0,
    0e0,
    0e0,
    -2.438808424736e-2,
    -1.850224608274e-2,
    -7.353811537540e-3,
    2.008547034175e-2,
    8.365454832702e-4,
    -8.947888514893e-4,
    -7.085843239142e-3,
    -1.455634327653e-2,
    -6.310912842359e-3,
    -7.124453943885e-3,
    1.166307407692e-2,
    5.542098698449e-3,
]

diff_eq = NBodyGravitationalEquation(3, masses)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: np.array(
        [pos * astronomical_unit for pos in positions_au]
        + [vel * astronomical_unit / day for vel in velocities_au_d]
    ),
)
ivp = InitialValueProblem(cp, (0.0, 5.0 * 365.0 * day), ic)

solver = ODEOperator("DOP853", day / 20.0)
solution = solver.solve(ivp)

for plot in solution.generate_plots(
    smallest_marker_size=2e-3,
    trajectory_line_width=0.15,
    span_scaling_factor=0.01,
):
    plot.show().close()
