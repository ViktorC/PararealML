import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pararealml import *
from pararealml.operators.ml.auto_regression import *
from pararealml.operators.fdm import *
from pararealml.utils.rand import SEEDS, set_random_seed

set_random_seed(SEEDS[0])

gamma = .01
diff_eq = CahnHilliardEquation(2, gamma=gamma)
mesh = Mesh([(0., 50.), (0., 50.)], [1., 1.])
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

fdm_op = FDMOperator(CrankNicolsonMethod(), diff, .01)
fdm_sol = fdm_op.solve(ivp)
fdm_sol_y = fdm_sol.discrete_y(fdm_op.vertex_oriented)
v_min = np.min(fdm_sol_y)
v_max = np.max(fdm_sol_y)
for i, plot in enumerate(fdm_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f'cahn_hilliard_fdm_{i}').close()

ar_op = AutoRegressionOperator(1.25, fdm_op.vertex_oriented)
ar_op.train(
    ivp,
    fdm_op,
    RandomForestRegressor(
        max_depth=24,
        n_estimators=240,
        n_jobs=4,
        verbose=True
    ),
    10,
    lambda t, y: y + np.random.normal(0., t / 375., size=y.shape)
)
ar_sol = ar_op.solve(ivp)
for i, plot in enumerate(ar_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f'cahn_hilliard_ar_{i}').close()
