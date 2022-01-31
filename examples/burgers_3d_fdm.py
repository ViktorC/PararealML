import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = BurgerEquation(3, 100)
mesh = Mesh(
    [(1., 5.), (0., 2. * np.pi), (.25 * np.pi, .75 * np.pi)],
    [.5, np.pi / 10., np.pi / 10.],
    CoordinateSystem.SPHERICAL
)
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 3)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 3)), is_static=True
        )
    )
] * 3
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(
    cp,
    lambda x: np.stack(
        [
            1. / x[:, 0] ** 2,
            np.zeros_like(x[:, 1]),
            np.zeros_like(x[:, 1])
        ],
        axis=-1
    )
)
ivp = InitialValueProblem(cp, (0., 100.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .5)
solution = solver.solve(ivp)

for plot in solution.generate_plots(quiver_scale=.1):
    plot.show().close()
