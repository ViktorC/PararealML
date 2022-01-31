import numpy as np

from pararealml import *
from pararealml.operators.fdm import *

gamma = .01
diff_eq = CahnHilliardEquation(2, gamma=gamma)
mesh = Mesh([(0., 10.), (0., 10.)], [.1, .1])
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True
        )
    )
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)

diff = ThreePointCentralDifferenceMethod()
y_0_0 = .05 * np.random.uniform(-1., 1., mesh.vertices_shape + (1,))
y_0_1 = y_0_0 ** 3 - y_0_0 - gamma * diff.laplacian(
    y_0_0, mesh, cp.create_boundary_constraints(True)[1][:, :1]
)
ic = DiscreteInitialCondition(
    cp,
    np.concatenate([y_0_0, y_0_1], axis=-1),
    True
)
ivp = InitialValueProblem(cp, (0., 5.), ic)

solver = FDMOperator(RK4(), diff, .0005)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
