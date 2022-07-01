import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

diff_eq = NavierStokesEquation(5000.0)
mesh = Mesh([(-2.5, 2.5), (0.0, 4.0)], [0.05, 0.05])
bcs = [
    (
        DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: [1.0, 0.1, None, None]),
            is_static=True,
        ),
        DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: [0.0, 0.0, None, None]),
            is_static=True,
        ),
    ),
    (
        DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: [0.0, 0.0, None, None]),
            is_static=True,
        ),
        DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: [0.0, 0.0, None, None]),
            is_static=True,
        ),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 4)))
ivp = InitialValueProblem(cp, (0.0, 100.0), ic)

solver = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.05)
solution = solver.solve(ivp)

for i, plot in enumerate(solution.generate_plots(quiver_scale=1.0)):
    plot.save(f"navier_stokes_{i}").show().close()
