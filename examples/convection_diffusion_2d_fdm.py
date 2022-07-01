import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = ConvectionDiffusionEquation(2, [2.0, 1.0])
mesh = Mesh([(0.0, 50.0), (0.0, 50.0)], [0.5, 0.5])
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp, [(np.array([12.5, 12.5]), np.array([[1.0, 0.0], [0.0, 1.0]]))], [100.0]
)
ivp = InitialValueProblem(cp, (0.0, 30.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.01)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
