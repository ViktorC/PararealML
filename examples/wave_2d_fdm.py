import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = WaveEquation(2)
mesh = Mesh([(-5.0, 5.0), (-5.0, 5.0)], [0.1, 0.1])
bcs = [
    (
        DirichletBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
        DirichletBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [(np.array([0.0, 2.5]), np.array([[0.1, 0.0], [0.0, 0.1]]))] * 2,
    [3.0, 0.0],
)
ivp = InitialValueProblem(cp, (0.0, 20.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.01)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
