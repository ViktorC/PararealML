import numpy as np
from matplotlib import cm

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = WaveEquation(2)
mesh = Mesh(
    [(2.5, 7.5), (0., 2 * np.pi)],
    [.1, np.pi / 100.],
    CoordinateSystem.POLAR
)
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        )
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [
        (
            np.array([-5., 0]),
            np.array([
                [.1, 0.],
                [0., .1]
            ])
        )
    ] * 2,
    [4., .0]
)
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .002)
solution = solver.solve(ivp)

for plot in solution.generate_plots(color_map=cm.coolwarm, equal_scale=True):
    plot.show().close()
