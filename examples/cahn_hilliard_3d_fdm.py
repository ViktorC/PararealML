import numpy as np

from pararealml import *
from pararealml.core.operators.fdm import *

diff_eq = CahnHilliardEquation(3, d=.25, gamma=.1)
mesh = Mesh([(0., 25.), (0., 25.), (0., 25.)], [.5, .5, .5])
bcs = (
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True)),
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True)),
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = DiscreteInitialCondition(
    cp,
    .05 * np.random.uniform(-1., 1., cp.y_shape(True)),
    True)
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
solution = solver.solve(ivp)
solution.plot('cahn_hilliard')
