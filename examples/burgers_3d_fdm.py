import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = BurgerEquation(3, 100)
mesh = Mesh(((0., 10.), (0., 10.), (0., 10.)), (1., 1., 1.))
bcs = (
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 3)), is_static=True),) * 2,
) * 3
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    (
        (
            np.array([5., 5., 5.]),
            np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ])
        ),
    ) * 3
)
ivp = InitialValueProblem(cp, (0., 100.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .5)
solver.solve(ivp).plot('3d_burgers')
