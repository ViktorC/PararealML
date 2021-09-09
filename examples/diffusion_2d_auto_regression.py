import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pararealml import *
from pararealml.core.operators.auto_regression import *
from pararealml.core.operators.fdm import *
from pararealml.utils.rand import SEEDS, set_random_seed

set_random_seed(SEEDS[0])

diff_eq = DiffusionEquation(2)
mesh = Mesh([(0., 10.), (0., 10.)], [.2, .2])
bcs = [
    (DirichletBoundaryCondition(
        lambda x, t: np.full((len(x), 1), 1.5), is_static=True),
     DirichletBoundaryCondition(
         lambda x, t: np.full((len(x), 1), 1.5), is_static=True)),
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True))
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [(np.array([5., 5.]), np.array([[2.5, 0.], [0., 2.5]]))],
    [100.]
)
ivp = InitialValueProblem(cp, (0., 2.), ic)

fdm_op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
ar_op = AutoRegressionOperator(.5, fdm_op.vertex_oriented)

fdm_sol = fdm_op.solve(ivp)
fdm_sol_y = fdm_sol.discrete_y(fdm_op.vertex_oriented)
v_min = np.min(fdm_sol_y)
v_max = np.max(fdm_sol_y)
fdm_sol.plot('diffusion_fdm', n_images=10, v_min=v_min, v_max=v_max)

ar_op.train(
    ivp,
    fdm_op,
    RandomForestRegressor(n_jobs=4, verbose=True),
    10,
    lambda t, y: y + np.random.normal(0., t / 3., size=y.shape)
)
ar_op.solve(ivp).plot('diffusion_ar', n_images=10, v_min=v_min, v_max=v_max)
