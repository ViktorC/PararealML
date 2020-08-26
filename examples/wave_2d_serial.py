import numpy as np

from pararealml import *

diff_eq = WaveEquation(2)
mesh = UniformGrid(((-5., 5.), (-5., 5.)), (.1, .1))
bcs = (
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0))),
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0)))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([0., 2.5]), np.array([[.1, 0.], [0., .1]])),) * 2,
    (3., .0))
ivp = InitialValueProblem(cp, (0., 50.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
solution = solver.solve(ivp)
solution.plot('wave_equation', n_images=50)
