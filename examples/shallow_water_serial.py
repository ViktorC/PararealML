import numpy as np

from pararealml import *

diff_eq = ShallowWaterEquation(.5)
mesh = UniformGrid(((-5., 5.), (-5., 5.)), (.1, .1))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (.0, None, None), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([2.5, 2.5]), np.array([[.25, 0.], [0., .25]])),) * 3,
    (1., .0, .0))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0025)
solution = solver.solve(ivp)
solution.plot('2d_shallow_water_equation')
