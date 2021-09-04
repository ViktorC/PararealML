import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pararealml import *
from pararealml.core.operators.auto_regression import *
from pararealml.core.operators.fdm import *
from pararealml.utils.rand import SEEDS, set_random_seed

set_random_seed(SEEDS[0])

diff_eq = CahnHilliardEquation(2, 1., .01)
mesh = Mesh([(0., 50.), (0., 50.)], [.5, .5])
bcs = (
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True)),
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 2)), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = DiscreteInitialCondition(
    cp,
    .05 * np.random.uniform(-1., 1., cp.y_shape(False)),
    False)
ivp = InitialValueProblem(cp, (0., 5.), ic)

fdm_op = FDMOperator(
    CrankNicolsonMethod(), ThreePointCentralFiniteDifferenceMethod(), .01)
ar_op = AutoRegressionOperator(1.25, fdm_op.vertex_oriented)

fdm_sol = fdm_op.solve(ivp)
fdm_sol_y = fdm_sol.discrete_y(fdm_op.vertex_oriented)
v_min = np.min(fdm_sol_y)
v_max = np.max(fdm_sol_y)
fdm_sol.plot('cahn_hilliard_fdm', n_images=10, v_min=v_min, v_max=v_max)

ar_op.train(
    ivp,
    fdm_op,
    RandomForestRegressor(max_depth=24, n_estimators=240, n_jobs=6, verbose=1),
    10,
    (0., .01))
ar_op.solve(ivp).plot(
    'cahn_hilliard_ar', n_images=10, v_min=v_min, v_max=v_max)
