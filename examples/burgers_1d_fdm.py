import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = BurgerEquation(1, 100)
mesh = Mesh([(0., 10.)], [.1])
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        )
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [(np.array([2.5]), np.array([[.5]]))]
)
ivp = InitialValueProblem(cp, (0., 200.), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .0025)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
