import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = WaveEquation(1)
mesh = Mesh([(-10., 10.)], [.1])
bcs = [
    (
        DirichletBoundaryCondition(
            lambda x, t: np.concatenate(
                [
                    np.full((len(x), 1), np.sin(t)),
                    np.full((len(x), 1), np.cos(t))
                ],
                axis=1
            )
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)),
            is_static=True
        )
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 2)))
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .005)
solution = solver.solve(ivp)

for plot in solution.generate_plots(equal_scale=True):
    plot.show().close()
