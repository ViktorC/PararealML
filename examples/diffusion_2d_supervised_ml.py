import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.supervised import *
from pararealml.utils.rand import SEEDS, set_random_seed

set_random_seed(SEEDS[0])

diff_eq = DiffusionEquation(2)
mesh = Mesh([(0.0, 10.0), (0.0, 10.0)], [1.0, 1.0])
bcs = [
    (
        DirichletBoundaryCondition(
            lambda x, t: np.full((len(x), 1), 1.5), is_static=True
        ),
        DirichletBoundaryCondition(
            lambda x, t: np.full((len(x), 1), 1.5), is_static=True
        ),
    ),
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp, [(np.array([5.0, 5.0]), np.array([[2.5, 0.0], [0.0, 2.5]]))], [100.0]
)
ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

fdm_op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.01)
fdm_sol = fdm_op.solve(ivp)
fdm_sol_y = fdm_sol.discrete_y(fdm_op.vertex_oriented)
v_min = np.min(fdm_sol_y)
v_max = np.max(fdm_sol_y)
for i, plot in enumerate(fdm_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f"diffusion_fdm_{i}").close()


def build_model(hidden_layer_size: int, optimizer: str, loss: str):
    regressor = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(hidden_layer_size, activation="tanh"),
            tf.keras.layers.Dense(diff_eq.y_dimension),
        ]
    )
    regressor.compile(optimizer=optimizer, loss=loss)
    return regressor


sml_op = SupervisedMLOperator(0.5, fdm_op.vertex_oriented)
sml_op.train(
    ivp,
    fdm_op,
    GridSearchCV(
        SKLearnKerasRegressor(build_model),
        {
            "hidden_layer_size": [10, 50, 100],
            "optimizer": ["adam"],
            "loss": ["mse"],
            "epochs": [100, 200, 500],
        },
        cv=5,
        verbose=5,
    ),
    10,
    lambda t, y: y + np.random.normal(0.0, t / 3.0, size=y.shape),
)
sml_sol = sml_op.solve(ivp)

joblib.dump(sml_op.model, "model.tar")

for i, plot in enumerate(sml_sol.generate_plots(v_min=v_min, v_max=v_max)):
    plot.save(f"diffusion_ar_{i}").close()
