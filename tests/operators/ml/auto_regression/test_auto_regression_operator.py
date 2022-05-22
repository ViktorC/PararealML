import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import optimizers

from pararealml.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import LotkaVolterraEquation, \
    LorenzEquation, WaveEquation, DiffusionEquation
from pararealml.initial_condition import ContinuousInitialCondition, \
    DiscreteInitialCondition, GaussianInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.operators.fdm.fdm_operator import FDMOperator
from pararealml.operators.fdm.numerical_differentiator import \
    ThreePointCentralDifferenceMethod
from pararealml.operators.fdm.numerical_integrator import RK4
from pararealml.operators.ml.auto_regression import AutoRegressionOperator, \
    SKLearnKerasRegressor
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.fnn_regressor import FNNRegressor
from pararealml.operators.ode.ode_operator import ODEOperator
from pararealml.utils.rand import set_random_seed


def perturbation_function(_: float, y: np.ndarray) -> np.ndarray:
    return y + np.random.normal(0., .01, size=y.shape)


def test_ar_operator_training_with_zero_iterations():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    oracle = ODEOperator('DOP853', .001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(
            ivp,
            oracle,
            LinearRegression(),
            0,
            perturbation_function)


def test_ar_operator_training_with_zero_n_jobs():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    oracle = ODEOperator('DOP853', .001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(
            ivp,
            oracle,
            LinearRegression(),
            100,
            perturbation_function,
            n_jobs=0)


def test_ar_operator_training_with_wrong_perturbed_initial_value_shape():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    oracle = ODEOperator('DOP853', .001)
    ar = AutoRegressionOperator(2.5, True)

    with pytest.raises(ValueError):
        ar.train(
            ivp,
            oracle,
            LinearRegression(),
            25,
            lambda t, y: np.array([1.]))


def test_ar_operator_data_generation_in_parallel():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = DiscreteInitialCondition(cp, np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    oracle = ODEOperator('DOP853', .001)
    ar = AutoRegressionOperator(2.5, True)

    inputs, targets = ar.generate_data(
        ivp, oracle, 20, perturbation_function, n_jobs=4)

    assert inputs.shape == (80, 3)
    assert targets.shape == (80, 3)


def test_ar_operator_on_ode():
    set_random_seed(0)

    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = ODEOperator('DOP853', .001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True)
    ar.train(
        ivp,
        oracle,
        RandomForestRegressor(),
        25,
        perturbation_function)
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 3)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .1


def test_ar_operator_on_ode_with_isolated_perturbations():
    set_random_seed(0)

    diff_eq = LotkaVolterraEquation(2., 1., .8, 1.)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1., 2.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = ODEOperator('DOP853', .001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True)
    ar.train(
        ivp,
        oracle,
        RandomForestRegressor(),
        25,
        perturbation_function,
        isolate_perturbations=True)
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 2)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .01


def test_ar_operator_on_ode_in_time_variant_mode():
    set_random_seed(0)

    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = ODEOperator('DOP853', .001)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True, time_variant=True)
    ar.train(
        ivp,
        oracle,
        RandomForestRegressor(),
        25,
        perturbation_function)
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 3)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .1


def test_ar_operator_on_pde():
    set_random_seed(0)

    diff_eq = WaveEquation(2)
    mesh = Mesh([(-5., 5.), (-5., 5.)], [1., 1.])
    bcs = [
        (DirichletBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True),
         DirichletBoundaryCondition(
             lambda x, t: np.zeros((len(x), 2)), is_static=True))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([0., 2.5]), np.array([[.1, 0.], [0., .1]]))] * 2,
        [3., .0]
    )
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True)
    ar.train(
        ivp,
        oracle,
        SKLearnKerasRegressor(
            DeepONet(
                np.prod(cp.y_shape(True)).item(),
                diff_eq.x_dimension,
                diff_eq.y_dimension * 10,
                diff_eq.y_dimension,
                [100, 50],
                [50, 50]
            ),
            optimizer=optimizers.Adam(
                learning_rate=optimizers.schedules.ExponentialDecay(
                        1e-2, decay_steps=500, decay_rate=.95
                )
            ),
            batch_size=968,
            epochs=500,
            max_predict_batch_size=300
        ),
        20,
        lambda t, y: y + np.random.normal(0., t / 75., size=y.shape))
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 11, 11, 2)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .5


def test_ar_operator_on_pde_in_time_variant_mode():
    set_random_seed(0)

    diff_eq = DiffusionEquation(1)
    mesh = Mesh([(0., 5.)], [1.])
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([2.5]), np.array([[.5]]))],
    )
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
    ref_solution = oracle.solve(ivp)

    ar = AutoRegressionOperator(2.5, True, time_variant=True)
    ar.train(
        ivp,
        oracle,
        SKLearnKerasRegressor(
            FNNRegressor([
                np.prod(cp.y_shape(True)).item() + diff_eq.x_dimension + 1,
                50,
                50,
                diff_eq.y_dimension
            ]),
            optimizer=optimizers.Adam(
                learning_rate=optimizers.schedules.ExponentialDecay(
                        1e-2, decay_steps=500, decay_rate=.95
                )
            ),
            batch_size=500,
            epochs=500,
        ),
        20,
        lambda t, y: y + np.random.normal(0., t / 75., size=y.shape))
    ml_solution = ar.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.discrete_y().shape == (4, 6, 1)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .01
