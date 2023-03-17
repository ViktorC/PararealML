import numpy as np
import pytest
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pararealml.boundary_condition import (
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    DiffusionEquation,
    LorenzEquation,
    LotkaVolterraEquation,
    PopulationGrowthEquation,
    WaveEquation,
)
from pararealml.initial_condition import (
    ContinuousInitialCondition,
    DiscreteInitialCondition,
    GaussianInitialCondition,
)
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.operators.fdm.fdm_operator import FDMOperator
from pararealml.operators.fdm.numerical_differentiator import (
    ThreePointCentralDifferenceMethod,
)
from pararealml.operators.fdm.numerical_integrator import RK4
from pararealml.operators.ml.auto_regression import (
    AutoRegressionOperator,
    SKLearnKerasRegressor,
)
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ode.ode_operator import ODEOperator
from pararealml.solution import Solution
from pararealml.utils.rand import set_random_seed


def perturbation_function(_: float, y: np.ndarray) -> np.ndarray:
    return y + np.random.normal(0.0, 0.01, size=y.shape)


def test_ar_operator_training_with_zero_iterations():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    oracle = ODEOperator("DOP853", 0.001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(ivp, oracle, LinearRegression(), 0, perturbation_function)


def test_ar_operator_training_with_zero_n_jobs():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    oracle = ODEOperator("DOP853", 0.001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(
            ivp,
            oracle,
            LinearRegression(),
            100,
            perturbation_function,
            n_jobs=0,
        )


def test_ar_operator_training_with_wrong_perturbed_initial_value_shape():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    oracle = ODEOperator("DOP853", 0.001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(
            ivp, oracle, LinearRegression(), 25, lambda t, y: np.array([1.0])
        )


def test_ar_operator_data_generation_in_parallel():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = DiscreteInitialCondition(cp, np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    oracle = ODEOperator("DOP853", 0.001)
    ar = AutoRegressionOperator(2.5, True)

    inputs, targets = ar.generate_data(
        ivp, oracle, 20, perturbation_function, n_jobs=4, seeds=list(range(4))
    )

    assert inputs.shape == (80, 3)
    assert targets.shape == (80, 3)


def test_ar_operator_data_generation_with_error_handling():
    set_random_seed(0)

    class ODEOperatorWithRandomError(ODEOperator):
        def solve(
            self, ivp: InitialValueProblem, parallel_enabled: bool = True
        ) -> Solution:
            if np.random.rand() > 0.5:
                raise RuntimeError
            return super(ODEOperatorWithRandomError, self).solve(
                ivp, parallel_enabled
            )

    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = DiscreteInitialCondition(cp, np.array([10.0]))
    test_ivp = InitialValueProblem(cp, (0.0, 20.0), ic)
    oracle = ODEOperatorWithRandomError("DOP853", 0.01)
    ar = AutoRegressionOperator(2.5, True)

    inputs, targets = ar.generate_data(
        test_ivp,
        oracle,
        20,
        perturbation_function,
        repeat_on_error=True,
        seeds=[0],
    )

    assert inputs.shape == (160, 1)
    assert targets.shape == (160, 1)


def test_ar_operator_on_ode():
    set_random_seed(0)

    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

    oracle = ODEOperator("DOP853", 0.001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True)
    ar.train(ivp, oracle, RandomForestRegressor(), 25, perturbation_function)
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 3)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10.0, 4))
    assert np.max(np.abs(diff.differences[0])) < 0.1


def test_ar_operator_on_ode_with_isolated_perturbations():
    set_random_seed(0)

    diff_eq = LotkaVolterraEquation(2.0, 1.0, 0.8, 1.0)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0, 2.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

    oracle = ODEOperator("DOP853", 0.001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True)
    ar.train(
        ivp,
        oracle,
        RandomForestRegressor(),
        25,
        perturbation_function,
        isolate_perturbations=True,
    )
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 2)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10.0, 4))
    assert np.max(np.abs(diff.differences[0])) < 0.01


def test_ar_operator_on_ode_in_time_variant_mode():
    set_random_seed(0)

    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

    oracle = ODEOperator("DOP853", 0.001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True, time_variant=True)
    ar.train(ivp, oracle, RandomForestRegressor(), 25, perturbation_function)
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 3)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10.0, 4))
    assert np.max(np.abs(diff.differences[0])) < 0.1


def test_ar_operator_on_pde():
    set_random_seed(0)

    diff_eq = WaveEquation(2)
    mesh = Mesh([(-5.0, 5.0), (-5.0, 5.0)], [1.0, 1.0])
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([0.0, 2.5]), np.array([[0.1, 0.0], [0.0, 0.1]]))] * 2,
        [3.0, 0.0],
    )
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

    oracle = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.1)
    ref_solution = oracle.solve(ivp)

    def build_model():
        model = DeepONet(
            branch_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(
                        np.prod(cp.y_shape(True)).item()
                    ),
                    tf.keras.layers.Dense(100, activation="tanh"),
                    tf.keras.layers.Dense(50, activation="tanh"),
                    tf.keras.layers.Dense(diff_eq.y_dimension * 10),
                ]
            ),
            trunk_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(diff_eq.x_dimension),
                    tf.keras.layers.Dense(50, activation="tanh"),
                    tf.keras.layers.Dense(50, activation="tanh"),
                    tf.keras.layers.Dense(diff_eq.y_dimension * 10),
                ]
            ),
            combiner_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(3 * diff_eq.y_dimension * 10),
                    tf.keras.layers.Dense(
                        diff_eq.y_dimension,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5),
                    ),
                ]
            ),
        )
        model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    1e-2, decay_steps=500, decay_rate=0.95
                )
            ),
            loss="mse",
        )
        return model

    ar = AutoRegressionOperator(2.5, True)
    ar.train(
        ivp,
        oracle,
        SKLearnKerasRegressor(
            build_model,
            batch_size=968,
            epochs=500,
            max_predict_batch_size=300,
        ),
        20,
        lambda t, y: y + np.random.normal(0.0, t / 75.0, size=y.shape),
    )
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 11, 11, 2)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10.0, 4))
    assert np.max(np.abs(diff.differences[0])) < 0.5


def test_ar_operator_on_pde_in_time_variant_mode():
    set_random_seed(0)

    diff_eq = DiffusionEquation(1)
    mesh = Mesh([(0.0, 5.0)], [1.0])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([2.5]), np.array([[0.5]]))],
    )
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)

    oracle = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.1)
    ref_solution = oracle.solve(ivp)

    def build_model():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation="tanh"),
                tf.keras.layers.Dense(50, activation="tanh"),
                tf.keras.layers.Dense(diff_eq.y_dimension),
            ]
        )
        model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    1e-2, decay_steps=500, decay_rate=0.95
                )
            ),
            loss="mse",
        )
        return model

    ar = AutoRegressionOperator(2.5, True, time_variant=True)
    ar.train(
        ivp,
        oracle,
        SKLearnKerasRegressor(
            build_model,
            batch_size=500,
            epochs=500,
        ),
        20,
        lambda t, y: y + np.random.normal(0.0, t / 75.0, size=y.shape),
    )
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 6, 1)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10.0, 4))
    assert np.max(np.abs(diff.differences[0])) < 0.01
