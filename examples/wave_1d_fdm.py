import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = WaveEquation(1)
mesh = Mesh([(-10., 10.)], [.1])
bcs = (
    (DirichletBoundaryCondition(lambda x, t: (np.sin(t), np.cos(t))),
     NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True)),
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda _: (0., 0.))
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .005)
solution = solver.solve(ivp)
solution.plot('1d_wave_equation', n_images=50)
