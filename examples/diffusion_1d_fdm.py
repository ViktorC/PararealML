import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = DiffusionEquation(1, 1.5)
mesh = Mesh([(0., 10.)], [.1])
bcs = [
    (
        NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        DirichletBoundaryCondition(lambda x, t: np.full((len(x), 1), t / 5.))
    )
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [(np.array([5.]), np.array([[.5]]))],
    [5.]
)
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .0025)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
