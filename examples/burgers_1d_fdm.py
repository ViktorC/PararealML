import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = BurgerEquation(1, 100)
mesh = Mesh(((0., 10.),), (.1,))
bcs = (
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True)),
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([2.5]), np.array([[1.]])),))
ivp = InitialValueProblem(cp, (0., 200.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0025)
solver.solve(ivp).plot('1d_burgers', n_images=40)
