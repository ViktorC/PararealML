import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = DiffusionEquation(1, 1.5)
mesh = Mesh([(0.0, 10.0)], [0.1])
bcs = [
    (
        NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        DirichletBoundaryCondition(lambda x, t: np.full((len(x), 1), t / 5.0)),
    )
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp, [(np.array([5.0]), np.array([[0.5]]))], [5.0]
)
ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.0025)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
