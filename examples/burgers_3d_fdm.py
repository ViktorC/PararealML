import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = BurgersEquation(3, 100)
mesh = Mesh(
    [(1.0, 5.0), (0.0, 2.0 * np.pi), (0.25 * np.pi, 0.75 * np.pi)],
    [0.5, np.pi / 10.0, np.pi / 10.0],
    CoordinateSystem.SPHERICAL,
)
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 3)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 3)), is_static=True
        ),
    )
] * 3
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(
    cp,
    lambda x: np.stack(
        [1.0 / x[:, 0] ** 2, np.zeros_like(x[:, 1]), np.zeros_like(x[:, 1])],
        axis=-1,
    ),
)
ivp = InitialValueProblem(cp, (0.0, 100.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.5)
solution = solver.solve(ivp)

for plot in solution.generate_plots(quiver_scale=0.1):
    plot.show().close()
