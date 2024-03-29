import numpy as np
from matplotlib import cm

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = ShallowWaterEquation(0.5)
mesh = Mesh([(-5.0, 5.0), (0.0, 5.0)], [0.1, 0.1])
bcs = [
    (
        NeumannBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.0, None, None)),
            is_static=True,
        ),
        NeumannBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.0, None, None)),
            is_static=True,
        ),
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [(np.array([2.5, 1.25]), np.array([[0.25, 0.0], [0.0, 0.25]]))] * 3,
    [1.0, 0.0, 0.0],
)
ivp = InitialValueProblem(cp, (0.0, 20.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.0025)
solution = solver.solve(ivp)

for plot in solution.generate_plots(color_map=cm.ocean):
    plot.show().close()
