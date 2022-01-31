import numpy as np

from pararealml import *
from pararealml.operators.ode import *

# Initial conditions taken from
# https://www.aanda.org/articles/aa/full/2002/08/aa1405/aa1405.right.html

astronomical_unit = 1.496e11
day = 24 * 3600

masses = [
    1.989e30,
    3.301e23,
    4.867e24,
    6.046e24,
    6.417e23,
    1.899e27,
    5.685e26,
    8.682e25,
    1.024e26,
    1.471e22,
]
positions_au = [
    0e0, 0e0, 0e0,
    -2.503321047836e-1, 1.873217481656e-1, 1.260230112145e-1,
    1.747780055994e-2, -6.624210296743e-1, -2.991203277122e-1,
    -9.091916173950e-1, 3.592925969244e-1, 1.557729610506e-1,
    1.203018828754e0, 7.270712989688e-1, 3.009561427569e-1,
    3.733076999471e0, 3.052424824299e0, 1.217426663570e0,
    6.164433062913e0, 6.366775402981e0, 2.364531109847e0,
    1.457964661868e1, -1.236891078519e1, -5.623617280033e0,
    1.695491139909e1, -2.288713988623e1, -9.789921035251e0,
    -9.707098450131e0, -2.804098175319e1, -5.823808919246e0
]
velocities_au_d = [
    0e0, 0e0, 0e0,
    -2.438808424736e-2, -1.850224608274e-2, -7.353811537540e-3,
    2.008547034175e-2, 8.365454832702e-4, -8.947888514893e-4,
    -7.085843239142e-3, -1.455634327653e-2, -6.310912842359e-3,
    -7.124453943885e-3, 1.166307407692e-2, 5.542098698449e-3,
    -5.086540617947e-3, 5.493643783389e-3, 2.478685100749e-3,
    -4.426823593779e-3, 3.394060157503e-3, 1.592261423092e-3,
    2.647505630327e-3, 2.487457379099e-3, 1.052000252243e-3,
    2.568651772461e-3, 1.681832388267e-3, 6.245613982833e-4,
    3.034112963576e-3, -1.111317562971e-3, -1.261841468083e-3
]

diff_eq = NBodyGravitationalEquation(3, masses)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: np.array(
        [pos * astronomical_unit for pos in positions_au] +
        [vel * astronomical_unit / day for vel in velocities_au_d]
    )
)
ivp = InitialValueProblem(cp, (0., 300. * 365. * day), ic)

solver = ODEOperator('DOP853', day / 2.)
solution = solver.solve(ivp)

for plot in solution.generate_plots(
    smallest_marker_size=1e-6,
    trajectory_line_width=.5
):
    plot.show().close()
