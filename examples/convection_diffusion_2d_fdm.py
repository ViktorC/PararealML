import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = ConvectionDiffusionEquation(2, [2., 1.])
mesh = Mesh([(0., 50.), (0., 50.)], [.5, .5])
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        )
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [
        (
            np.array([12.5, 12.5]),
            np.array([
                [1., 0.],
                [0., 1.]
            ])
        )
    ],
    [100.]
)
ivp = InitialValueProblem(cp, (0., 30.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .01)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
