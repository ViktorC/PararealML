import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = WaveEquation(2)
mesh = Mesh([(-5., 5.), (-5., 5.)], [.1, .1])
bcs = [
    (DirichletBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     DirichletBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True))
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [
        (
            np.array([0., 2.5]),
            np.array([
                [.1, 0.],
                [0., .1]
            ])
        )
    ] * 2,
    [3., .0]
)
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
solution = solver.solve(ivp)
solution.plot('2d_wave_equation', n_images=50)
