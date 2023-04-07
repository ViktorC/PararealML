import numpy as np
import pytest
import tensorflow as tf

from pararealml import SymbolicEquationSystem
from pararealml.boundary_condition import (
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    DifferentialEquation,
    DiffusionEquation,
    LotkaVolterraEquation,
    NavierStokesEquation,
    PopulationGrowthEquation,
    WaveEquation,
)
from pararealml.initial_condition import (
    ContinuousInitialCondition,
    MarginalBetaProductInitialCondition,
    vectorize_ic_function,
)
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import CoordinateSystem, Mesh
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    UniformRandomCollocationPointSampler,
)
from pararealml.operators.ml.physics_informed.physics_informed_ml_operator import (  # noqa: 501
    DataArgs,
    ModelArgs,
    OptimizationArgs,
    PhysicsInformedMLOperator,
)
from pararealml.utils.rand import set_random_seed


def test_piml_operator_on_ode_with_analytic_solution():
    set_random_seed(0)

    r = 4.0
    y_0 = 1.0

    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0.0, 0.25)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=100,
            n_batches=5,
            n_ic_repeats=5,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(
                            50,
                            activation="tanh",
                        )
                        for _ in range(3)
                    ]
                    + [
                        tf.keras.layers.Dense(
                            1,
                            activation="tanh",
                        )
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(
                            50,
                            activation="tanh",
                        )
                        for _ in range(3)
                    ]
                    + [
                        tf.keras.layers.Dense(
                            1,
                            activation="tanh",
                        )
                    ]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(3),
                        tf.keras.layers.Dense(
                            50,
                            activation="tanh",
                        ),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            )
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    1e-2, decay_steps=100, decay_rate=0.9
                )
            ),
            epochs=500,
        ),
    )

    assert len(history.epoch) == 500
    assert history.history["loss"][-1] < 2e-4
    assert test_loss is None

    ivp = InitialValueProblem(
        cp,
        t_interval,
        ic,
        lambda _ivp, t, x: np.array([y_0 * np.e ** (r * t)]),
    )

    solution = piml.solve(ivp)

    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (250, 1)

    analytic_y = np.array([ivp.exact_y(t) for t in solution.t_coordinates])

    assert np.mean(np.abs(analytic_y - solution.discrete_y())) < 4e-3
    assert np.max(np.abs(analytic_y - solution.discrete_y())) < 5e-3


def test_piml_operator_on_ode_system():
    set_random_seed(0)

    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0.0, 0.5)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.01, True)

    training_y_0_functions = [
        lambda _: np.array([47.5, 25.0]),
        lambda _: np.array([47.5, 27.5]),
        lambda _: np.array([50.0, 22.5]),
        lambda _: np.array([50.0, 27.5]),
        lambda _: np.array([52.5, 22.5]),
        lambda _: np.array([52.5, 25.0]),
    ]
    test_y_0_functions = [
        lambda _: np.array([47.5, 22.5]),
        lambda _: np.array([50.0, 25.0]),
        lambda _: np.array([52.5, 27.5]),
    ]

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_batches=3,
        ),
        validation_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=20,
            n_batches=1,
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions, n_domain_points=20, n_batches=1
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(20, activation="tanh")
                        for _ in range(4)
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(20, activation="tanh")
                        for _ in range(4)
                    ]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(60),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
            ic_loss_weight=2.0,
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                "class_name": "Adam",
                "config": {"learning_rate": 1e-4},
            },
            epochs=3,
        ),
    )

    assert len(test_loss) == 10
    assert len(training_history.epoch) == 3
    for i in range(2):
        assert (
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )
        assert (
            training_history.history["val_loss"][i + 1]
            < training_history.history["val_loss"][i]
        )

    ic = ContinuousInitialCondition(cp, lambda _: np.array([50.0, 25.0]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = piml.solve(ivp)
    assert solution.d_t == 0.01
    assert solution.discrete_y().shape == (50, 2)


def test_piml_operator_on_pde_with_dynamic_boundary_conditions():
    set_random_seed(0)

    diff_eq = DiffusionEquation(1, 0.25)
    mesh = Mesh([(0.0, 1.0)], (0.1,))
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.full((len(x), 1), t)),
            NeumannBoundaryCondition(lambda x, t: np.full((len(x), 1), t)),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 0.5)

    training_y_0_functions = [
        MarginalBetaProductInitialCondition(cp, [[(p, p)]]).y_0
        for p in [2.0, 3.0, 4.0, 5.0]
    ]
    validation_y_0_functions = [
        MarginalBetaProductInitialCondition(cp, [[(p, p)]]).y_0
        for p in [2.5, 3.5, 4.5]
    ]

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_boundary_points=20,
            n_batches=2,
        ),
        validation_data_args=DataArgs(
            y_0_functions=validation_y_0_functions,
            n_domain_points=25,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(50, activation="tanh")
                        for _ in range(3)
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(50, activation="tanh")
                        for _ in range(3)
                    ]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(150),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
            ic_loss_weight=10.0,
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                "class_name": "Adam",
                "config": {"learning_rate": 1e-4},
            },
            epochs=3,
        ),
    )

    assert len(training_history.epoch) == 3
    assert not test_loss
    for i in range(2):
        assert (
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )
        assert (
            training_history.history["val_loss"][i + 1]
            < training_history.history["val_loss"][i]
        )

    ic = MarginalBetaProductInitialCondition(cp, [[(3.5, 3.5)]])
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = piml.solve(ivp)
    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (500, 11, 1)


def test_piml_operator_on_pde_system():
    set_random_seed(0)

    diff_eq = NavierStokesEquation()
    mesh = Mesh([(-2.5, 2.5), (0.0, 4.0)], [1.0, 1.0])
    ic_function = vectorize_ic_function(
        lambda x: [
            2.0 * x[0] - 4.0,
            2.0 * x[0] ** 2 + 3.0 * x[1] - x[0] * x[1] ** 2,
            4.0 * x[0] - x[1] ** 2,
            2.0 * x[0] * x[1] - 3.0,
        ]
    )
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: ic_function(x), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: ic_function(x), is_static=True
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, ic_function)
    t_interval = (0.0, 0.5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(
                            50,
                            kernel_initializer="he_uniform",
                            activation="softplus",
                        )
                        for _ in range(2)
                    ]
                    + [
                        tf.keras.layers.Dense(
                            25, kernel_initializer="he_uniform"
                        )
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(
                            50,
                            kernel_initializer="he_uniform",
                            activation="softplus",
                        )
                        for _ in range(2)
                    ]
                    + [
                        tf.keras.layers.Dense(
                            25, kernel_initializer="he_uniform"
                        )
                    ]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(75),
                        tf.keras.layers.Dense(
                            25,
                            kernel_initializer="he_uniform",
                            activation="softplus",
                        ),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            )
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=5e-8), epochs=3
        ),
    )

    assert len(training_history.epoch) == 3
    for i in range(2):
        assert np.all(
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )

    solution = piml.solve(ivp)
    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (500, 6, 5, 4)


def test_piml_operator_on_pde_with_t_and_x_dependent_rhs():
    class TestDiffEq(DifferentialEquation):
        def __init__(self):
            super(TestDiffEq, self).__init__(2, 1)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem(
                [
                    self.symbols.t
                    / 100.0
                    * (self.symbols.x[0] + self.symbols.x[1]) ** 2
                ]
            )

    diff_eq = TestDiffEq()
    mesh = Mesh([(-1.0, 1.0), (0.0, 2.0)], [2.0, 1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 1)))
    t_interval = (0.0, 1.0)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.05, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(60),
                        tf.keras.layers.Dense(
                            diff_eq.y_dimension, kernel_regularizer="l2"
                        ),
                    ]
                ),
            ),
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=2e-5), epochs=3
        ),
    )

    assert len(training_history.epoch) == 3
    for i in range(2):
        assert np.all(
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )

    solution = piml.solve(ivp)
    assert solution.d_t == 0.05
    assert solution.discrete_y().shape == (20, 2, 3, 1)


def test_piml_operator_on_polar_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(2)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2 * np.pi)],
        [2.0, np.pi / 5.0],
        CoordinateSystem.POLAR,
    )
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.ones((len(x), 1)), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: np.full((len(x), 1), 1.0 / 11.0), is_static=True
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
    ic = ContinuousInitialCondition(cp, lambda x: 1.0 / x[:, :1])
    t_interval = (0.0, 0.5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(60),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=2e-5), epochs=3
        ),
    )

    assert len(training_history.epoch) == 3
    for i in range(2):
        assert np.all(
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )

    solution = piml.solve(ivp)
    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (500, 6, 11, 1)


def test_piml_operator_on_cylindrical_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2 * np.pi), (0.0, 2.0)],
        [2.0, np.pi / 5.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.ones((len(x), 1)), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: np.full((len(x), 1), 1.0 / 11.0), is_static=True
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
    ic = ContinuousInitialCondition(cp, lambda x: 1.0 / x[:, :1])
    t_interval = (0.0, 0.5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(60),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=2e-5), epochs=3
        ),
    )

    assert len(training_history.epoch) == 3
    for i in range(2):
        assert np.all(
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )

    solution = piml.solve(ivp)
    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (500, 6, 11, 3, 1)


def test_piml_operator_on_spherical_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2 * np.pi), (0.25 * np.pi, 0.75 * np.pi)],
        [2.0, np.pi / 5.0, np.pi / 4],
        CoordinateSystem.SPHERICAL,
    )
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.ones((len(x), 1)), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: np.full((len(x), 1), 1.0 / 11.0), is_static=True
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
    ic = ContinuousInitialCondition(cp, lambda x: 1.0 / x[:, :1])
    t_interval = (0.0, 0.5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    training_history, test_loss = piml.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(30, activation="tanh")
                        for _ in range(2)
                    ]
                    + [tf.keras.layers.Dense(20, activation="tanh")]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(60),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=2e-5), epochs=3
        ),
    )

    assert len(training_history.epoch) == 3
    for i in range(2):
        assert np.all(
            training_history.history["loss"][i + 1]
            < training_history.history["loss"][i]
        )

    solution = piml.solve(ivp)
    assert solution.d_t == 0.001
    assert solution.discrete_y().shape == (500, 6, 11, 3, 1)


def test_piml_operator_with_no_model_training_without_model_args():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0.0, 1.0)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0]))

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.001, True)

    with pytest.raises(ValueError):
        piml.train(
            cp,
            t_interval,
            training_data_args=DataArgs(
                y_0_functions=[ic.y_0],
                n_domain_points=25,
                n_batches=5,
                n_ic_repeats=5,
            ),
            optimization_args=OptimizationArgs(
                optimizer=tf.optimizers.SGD(), epochs=100
            ),
        )


def test_piml_operator_in_ar_mode_training_with_invalid_t_interval():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0.0, 1.0)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0]))

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.25, True, auto_regressive=True)

    with pytest.raises(ValueError):
        piml.train(
            cp,
            t_interval,
            training_data_args=DataArgs(
                y_0_functions=[ic.y_0], n_domain_points=50, n_batches=1
            ),
            model_args=ModelArgs(
                model=DeepONet(
                    branch_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(
                                np.prod(cp.y_vertices_shape).item()
                            ),
                            tf.keras.layers.Dense(1, activation="tanh"),
                        ]
                    ),
                    trunk_net=tf.keras.Sequential(
                        [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                        + [
                            tf.keras.layers.Dense(50, activation="tanh")
                            for _ in range(3)
                        ]
                        + [tf.keras.layers.Dense(1, activation="tanh")]
                    ),
                    combiner_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(3),
                            tf.keras.layers.Dense(diff_eq.y_dimension),
                        ]
                    ),
                ),
            ),
            optimization_args=OptimizationArgs(
                optimizer={"class_name": "Adam"}, epochs=100
            ),
        )


def test_piml_operator_in_ar_mode_training_with_diff_eq_containing_t_term():
    class TestDiffEq(DifferentialEquation):
        def __init__(self):
            super(TestDiffEq, self).__init__(0, 1)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem([self.symbols.t])

    diff_eq = TestDiffEq()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0]))

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.25, True, auto_regressive=True)

    with pytest.raises(ValueError):
        piml.train(
            cp,
            (0.0, 0.25),
            training_data_args=DataArgs(
                y_0_functions=[ic.y_0], n_domain_points=50, n_batches=1
            ),
            model_args=ModelArgs(
                model=DeepONet(
                    branch_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(
                                np.prod(cp.y_vertices_shape).item()
                            ),
                            tf.keras.layers.Dense(1, activation="tanh"),
                        ]
                    ),
                    trunk_net=tf.keras.Sequential(
                        [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                        + [
                            tf.keras.layers.Dense(50, activation="tanh")
                            for _ in range(3)
                        ]
                        + [tf.keras.layers.Dense(1, activation="tanh")]
                    ),
                    combiner_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(3),
                            tf.keras.layers.Dense(diff_eq.y_dimension),
                        ]
                    ),
                ),
            ),
            optimization_args=OptimizationArgs(
                optimizer={"class_name": "Adam"}, epochs=100
            ),
        )


def test_piml_operator_in_ar_mode_training_with_dynamic_boundary_conditions():
    diff_eq = DiffusionEquation(1, 0.25)
    mesh = Mesh([(0.0, 0.5)], (0.05,))
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.full((len(x), 1), t), is_static=False
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 0.5)
    y_0_functions = [lambda x: np.zeros((len(x), 1))]

    piml = PhysicsInformedMLOperator(
        UniformRandomCollocationPointSampler(), 0.001, True
    )

    with pytest.raises(ValueError):
        piml.train(
            cp,
            t_interval,
            training_data_args=DataArgs(
                y_0_functions=y_0_functions,
                n_domain_points=50,
                n_boundary_points=20,
                n_batches=2,
            ),
            model_args=ModelArgs(
                model=DeepONet(
                    branch_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(
                                np.prod(cp.y_vertices_shape).item()
                            )
                        ]
                        + [
                            tf.keras.layers.Dense(50, activation="tanh")
                            for _ in range(3)
                        ]
                    ),
                    trunk_net=tf.keras.Sequential(
                        [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                        + [
                            tf.keras.layers.Dense(50, activation="tanh")
                            for _ in range(3)
                        ]
                    ),
                    combiner_net=tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(150),
                            tf.keras.layers.Dense(diff_eq.y_dimension),
                        ]
                    ),
                ),
                ic_loss_weight=10.0,
            ),
            optimization_args=OptimizationArgs(
                optimizer={"class_name": "Adam"}, epochs=3
            ),
        )


def test_piml_operator_in_ar_mode_on_ode():
    set_random_seed(0)

    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0.0, 1.0)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(sampler, 0.25, True, auto_regressive=True)

    assert piml.auto_regressive

    piml.train(
        cp,
        (0.0, 0.25),
        training_data_args=DataArgs(
            y_0_functions=[
                lambda _, _y_0=y_0: np.array([_y_0])
                for y_0 in np.linspace(0.5, 2.0, 10)
            ],
            n_domain_points=50,
            n_batches=1,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_vertices_shape).item()
                        ),
                        tf.keras.layers.Dense(1, activation="tanh"),
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(50, activation="tanh")
                        for _ in range(3)
                    ]
                    + [tf.keras.layers.Dense(1, activation="tanh")]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(3),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                "class_name": "Adam",
                "config": {
                    "learning_rate": tf.optimizers.schedules.ExponentialDecay(
                        1e-2, decay_steps=25, decay_rate=0.95
                    )
                },
            },
            epochs=5,
        ),
    )

    sol = piml.solve(ivp)
    assert np.allclose(sol.t_coordinates, [0.25, 0.5, 0.75, 1.0])
    assert sol.discrete_y().shape == (4, 1)


def test_piml_operator_in_ar_mode_on_pde():
    set_random_seed(0)

    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 1.0)], (0.2,))
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 1.0)
    ic = MarginalBetaProductInitialCondition(cp, [[(3.5, 3.5)], [(3.5, 3.5)]])
    ivp = InitialValueProblem(cp, t_interval, ic)

    training_y_0_functions = [
        MarginalBetaProductInitialCondition(cp, [[(p, p)], [(p, p)]]).y_0
        for p in [2.0, 3.0, 4.0, 5.0]
    ]
    sampler = UniformRandomCollocationPointSampler()
    piml = PhysicsInformedMLOperator(
        sampler, 0.25, False, auto_regressive=True
    )

    assert piml.auto_regressive

    piml.train(
        cp,
        (0.0, 0.25),
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_boundary_points=20,
            n_batches=2,
        ),
        model_args=ModelArgs(
            model=DeepONet(
                branch_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            np.prod(cp.y_cells_shape).item()
                        )
                    ]
                    + [
                        tf.keras.layers.Dense(50, activation="tanh")
                        for _ in range(3)
                    ]
                ),
                trunk_net=tf.keras.Sequential(
                    [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
                    + [
                        tf.keras.layers.Dense(50, activation="tanh")
                        for _ in range(3)
                    ]
                ),
                combiner_net=tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(150),
                        tf.keras.layers.Dense(diff_eq.y_dimension),
                    ]
                ),
            ),
            diff_eq_loss_weight=[2.0, 1.0],
            ic_loss_weight=10.0,
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(learning_rate=1e-4), epochs=2
        ),
    )

    sol = piml.solve(ivp)
    assert np.allclose(sol.t_coordinates, [0.25, 0.5, 0.75, 1.0])
    assert sol.discrete_y().shape == (4, 5, 2)
