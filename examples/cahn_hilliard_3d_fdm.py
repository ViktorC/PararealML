import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

gamma = 0.5
diff_eq = CahnHilliardEquation(3, gamma=gamma)
mesh = Mesh(
    [(1.0, 31.0), (1.0, 31.0), (1.0, 31.0)],
    [1.0, 1.0, 1.0],
)
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
    )
] * 3
cp = ConstrainedProblem(diff_eq, mesh, bcs)

diff = ThreePointCentralDifferenceMethod()
y_0_0 = 0.05 * np.random.uniform(-1.0, 1.0, mesh.vertices_shape + (1,))
y_0_1 = (
    y_0_0**3
    - y_0_0
    - gamma
    * diff.laplacian(
        y_0_0, mesh, cp.create_boundary_constraints(True)[1][:, :1]
    )
)
ic = DiscreteInitialCondition(
    cp, np.concatenate([y_0_0, y_0_1], axis=-1), True
)
ivp = InitialValueProblem(cp, (0.0, 150.0), ic)

solver = FDMOperator(RK4(), diff, 0.05)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
