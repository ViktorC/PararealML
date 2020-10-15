import numpy as np

from pararealml import *

diff_eq = ConvectionDiffusionEquation(2, [2., 1.])
mesh = UniformGrid(((0., 50.), (0., 50.)), (.5, .5))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([12.5, 12.5]), np.array([[10., 0.], [0., 10.]])),), (100.,))
ivp = InitialValueProblem(cp, (0., 30.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
solver.solve(ivp).plot('2d_convection_diffusion', n_images=20, three_d=True)
