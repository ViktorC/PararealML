import numpy as np

from pararealml import *

diff_eq = CahnHilliardEquation(2)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.1, .1))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = DiscreteInitialCondition(
    cp,
    .05 * np.random.uniform(-1., 1., cp.y_shape(True)),
    True)
ivp = InitialValueProblem(cp, (0., 7.5), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .0005)
solution = solver.solve(ivp)
solution.plot('cahn_hilliard')
