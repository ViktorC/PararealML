import numpy as np
from fipy import LinearLUSolver

from pararealml import *

diff_eq = DiffusionEquation(1, 1.5)
mesh = UniformGrid(((0., 10.),), (.1,))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True)),
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([5.]), np.array([[2.5]])),),
    (20.,))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FVMOperator(LinearLUSolver(), .01)
solver.solve(ivp).plot('1d_diffusion')
