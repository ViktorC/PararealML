import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = ShallowWaterEquation(.5)
mesh = Mesh(
    [(1., 11.), (0., 2 * np.pi)],
    [.2, np.pi / 50.],
    CoordinateSystem.POLAR)
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([-6., 0.]), np.array([[.25, 0.], [0., .25]])),) * 3,
    (1., .0, .0))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0025)
solution = solver.solve(ivp)
solution.plot('polar_shallow_water_equation')
