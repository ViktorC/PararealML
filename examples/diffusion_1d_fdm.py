import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = DiffusionEquation(1, 1.5)
mesh = Mesh(((0., 10.),), (.1,))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (0.,)),
     DirichletBoundaryCondition(lambda x, t: (t / 5.,))),
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([5.]), np.array([[2.5]])),),
    (20.,))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0025)
solver.solve(ivp).plot('1d_diffusion')
