import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = WaveEquation(2)
mesh = Mesh(
    [(2.5, 7.5), (0., 2 * np.pi)],
    [.1, np.pi / 100.],
    CoordinateSystem.POLAR)
bcs = [
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True))
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
    [1., .0])
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .002)
solution = solver.solve(ivp)
solution.plot('polar_wave_equation', n_images=50)
