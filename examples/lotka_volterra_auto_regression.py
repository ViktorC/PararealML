import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pararealml import *
from pararealml.operators.ode import *
from pararealml.operators.ml.auto_regression import *
from pararealml.utils.rand import SEEDS, set_random_seed

set_random_seed(SEEDS[0])

diff_eq = LotkaVolterraEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.array([100., 15.]))
ivp = InitialValueProblem(cp, (0., 10.), ic)

ode_op = ODEOperator('DOP853', 1e-3)
ode_sol = ode_op.solve(ivp)
ode_sol_y = ode_sol.discrete_y()
v_min = np.min(ode_sol_y)
v_max = np.max(ode_sol_y)
for i, plot in enumerate(ode_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f'lotka_volterra_ode_{i}').close()

ar_op = AutoRegressionOperator(.1, ode_op.vertex_oriented)
ar_op.train(
    ivp,
    ode_op,
    RandomForestRegressor(n_estimators=250, n_jobs=4, verbose=True),
    50,
    lambda t, y: y + np.random.normal(0., t / 99., size=y.shape),
    isolate_perturbations=True
)
ar_sol = ar_op.solve(ivp)
for i, plot in enumerate(ar_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f'lotka_volterra_ar_{i}').close()
