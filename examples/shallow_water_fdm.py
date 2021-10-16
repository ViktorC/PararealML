import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = ShallowWaterEquation(.5)
mesh = Mesh([(-5., 5.), (-5., 5.)], [.1, .1])
bcs = [
    (NeumannBoundaryCondition(
        vectorize_bc_function(lambda x, t: (.0, None, None)),
        is_static=True),
     NeumannBoundaryCondition(
         vectorize_bc_function(lambda x, t: (.0, None, None)),
         is_static=True))
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [
        (
            np.array([2.5, 2.5]),
            np.array([
                [.25, 0.],
                [0., .25]
            ])
        )
    ] * 3,
    [1., .0, .0]
)
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .0025)
solution = solver.solve(ivp)
solution.plot('2d_shallow_water_equation')
