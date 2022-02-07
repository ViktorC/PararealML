import numpy as np
import pytest
from tensorflow import optimizers

from pararealml import SymbolicEquationSystem
from pararealml.boundary_condition import NeumannBoundaryCondition, \
    DirichletBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import DifferentialEquation, \
    PopulationGrowthEquation, LotkaVolterraEquation, DiffusionEquation, \
    WaveEquation, NavierStokesEquation
from pararealml.initial_condition import ContinuousInitialCondition, \
    vectorize_ic_function, BetaInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh, CoordinateSystem
from pararealml.operators.ml.pidon.collocation_point_sampler import \
    UniformRandomCollocationPointSampler
from pararealml.operators.ml.pidon.pidon_operator import PIDONOperator, \
    DataArgs, ModelArgs, OptimizationArgs, SecondaryOptimizationArgs
from pararealml.utils.rand import set_random_seed


def test_pidon_operator_on_ode_with_analytic_solution():
    set_random_seed(0)

    r = 4.
    y_0 = 1.

    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., .25)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=25,
            n_batches=5,
            n_ic_repeats=5
        ),
        model_args=ModelArgs(
            latent_output_size=1,
            trunk_hidden_layer_sizes=[50, 50, 50],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.SGD(),
            epochs=100,
            verbose=False
        ),
        secondary_optimization_args=SecondaryOptimizationArgs(
            max_iterations=100,
            verbose=False
        )
    )

    assert len(training_loss_history) == 101
    assert len(test_loss_history) == 0
    assert training_loss_history[-1].weighted_total_loss.numpy() < 5e-5

    ivp = InitialValueProblem(
        cp,
        t_interval,
        ic,
        lambda _ivp, t, x: np.array([y_0 * np.e ** (r * t)])
    )

    solution = pidon.solve(ivp)

    assert solution.d_t == .001
    assert solution.discrete_y().shape == (250, 1)

    analytic_y = np.array([ivp.exact_y(t) for t in solution.t_coordinates])

    assert np.mean(np.abs(analytic_y - solution.discrete_y())) < 1e-3
    assert np.max(np.abs(analytic_y - solution.discrete_y())) < 2.5e-3


def test_pidon_operator_on_ode_system():
    set_random_seed(0)

    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., .5)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .01, True)

    training_y_0_functions = [
        lambda _: np.array([47.5, 25.]),
        lambda _: np.array([47.5, 27.5]),
        lambda _: np.array([50., 22.5]),
        lambda _: np.array([50., 27.5]),
        lambda _: np.array([52.5, 22.5]),
        lambda _: np.array([52.5, 25.]),
    ]
    test_y_0_functions = [
        lambda _: np.array([47.5, 22.5]),
        lambda _: np.array([50., 25.]),
        lambda _: np.array([52.5, 27.5])
    ]

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_batches=3
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=20,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[20, 20, 20],
            trunk_hidden_layer_sizes=[20, 20, 20],
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                'class_name': 'Adam',
                'config': {
                    'learning_rate': 1e-4
                }
            },
            epochs=3,
            ic_loss_weight=2.,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    assert len(test_loss_history) == 3
    for i in range(2):
        assert np.sum(
            training_loss_history[i + 1].weighted_total_loss.numpy()) < \
               np.sum(training_loss_history[i].weighted_total_loss.numpy())
        assert np.sum(test_loss_history[i + 1].weighted_total_loss.numpy()) < \
            np.sum(test_loss_history[i].weighted_total_loss.numpy())

    ic = ContinuousInitialCondition(cp, lambda _: np.array([50., 25.]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = pidon.solve(ivp)
    assert solution.d_t == .01
    assert solution.discrete_y().shape == (50, 2)


def test_pidon_operator_on_pde_with_dynamic_boundary_conditions():
    set_random_seed(0)

    diff_eq = DiffusionEquation(1, .25)
    mesh = Mesh([(0., 1.)], (.1,))
    bcs = [
        (NeumannBoundaryCondition(lambda x, t: np.full((len(x), 1), t)),
         NeumannBoundaryCondition(lambda x, t: np.full((len(x), 1), t))),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., .5)

    training_y_0_functions = [
        BetaInitialCondition(cp, [(p, p)]).y_0 for p in [2., 3., 4., 5.]
    ]
    test_y_0_functions = [
        BetaInitialCondition(cp, [(p, p)]).y_0 for p in [2.5, 3.5, 4.5]
    ]

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_boundary_points=20,
            n_batches=2
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=25,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=50,
            branch_hidden_layer_sizes=[50, 50],
            trunk_hidden_layer_sizes=[50, 50],
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                'class_name': 'Adam',
                'config': {
                    'learning_rate': 1e-4
                }
            },
            epochs=3,
            ic_loss_weight=10.,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    assert len(test_loss_history) == 3
    for i in range(2):
        assert training_loss_history[i + 1].weighted_total_loss.numpy() < \
            training_loss_history[i].weighted_total_loss.numpy()
        assert test_loss_history[i + 1].weighted_total_loss.numpy() < \
            test_loss_history[i].weighted_total_loss.numpy()

    ic = BetaInitialCondition(cp, [(3.5, 3.5)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 11, 1)


def test_pidon_operator_on_pde_system():
    set_random_seed(0)

    diff_eq = NavierStokesEquation()
    mesh = Mesh([(-2.5, 2.5), (0., 4.)], [1., 1.])
    ic_function = vectorize_ic_function(lambda x: [
        2. * x[0] - 4.,
        2. * x[0] ** 2 + 3. * x[1] - x[0] * x[1] ** 2,
        4. * x[0] - x[1] ** 2,
        2. * x[0] * x[1] - 3.
    ])
    bcs = [
        (DirichletBoundaryCondition(
            lambda x, t: ic_function(x),
            is_static=True),
         DirichletBoundaryCondition(
             lambda x, t: ic_function(x),
             is_static=True))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, ic_function)
    t_interval = (0., .5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[20, 20],
            trunk_hidden_layer_sizes=[20, 20],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            epochs=3,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    for i in range(2):
        assert np.all(
            training_loss_history[i + 1].weighted_total_loss.numpy() <
            training_loss_history[i].weighted_total_loss.numpy())

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 6, 5, 4)


def test_fdm_operator_on_pde_with_t_and_x_dependent_rhs():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(2, 1)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem([
                self.symbols.t / 100. *
                (self.symbols.x[0] + self.symbols.x[1]) ** 2
            ])

    diff_eq = TestDiffEq()
    mesh = Mesh([(-1., 1.), (0., 2.)], [2., 1.])
    bcs = [
        (NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
         NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 1)))
    t_interval = (0., 1.)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .05, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[30, 30],
            trunk_hidden_layer_sizes=[30, 30],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=2e-5),
            epochs=3,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    for i in range(2):
        assert np.all(
            training_loss_history[i + 1].weighted_total_loss.numpy() <
            training_loss_history[i].weighted_total_loss.numpy())

    solution = pidon.solve(ivp)
    assert solution.d_t == .05
    assert solution.discrete_y().shape == (20, 2, 3, 1)


def test_pidon_operator_on_polar_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(2)
    mesh = Mesh(
        [(1., 11.), (0., 2 * np.pi)],
        [2., np.pi / 5.],
        CoordinateSystem.POLAR)
    bcs = [
        (DirichletBoundaryCondition(
            lambda x, t: np.ones((len(x), 1)), is_static=True),
         DirichletBoundaryCondition(
             lambda x, t: np.full((len(x), 1), 1. / 11.), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: 1. / x[:, :1])
    t_interval = (0., .5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[30, 30],
            trunk_hidden_layer_sizes=[30, 30],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=2e-5),
            epochs=3,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    for i in range(2):
        assert np.all(
            training_loss_history[i + 1].weighted_total_loss.numpy() <
            training_loss_history[i].weighted_total_loss.numpy())

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 6, 11, 1)


def test_pidon_operator_on_cylindrical_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1., 11.), (0., 2 * np.pi), (0., 2.)],
        [2., np.pi / 5., 1.],
        CoordinateSystem.CYLINDRICAL)
    bcs = [
        (DirichletBoundaryCondition(
            lambda x, t: np.ones((len(x), 1)), is_static=True),
         DirichletBoundaryCondition(
             lambda x, t: np.full((len(x), 1), 1. / 11.), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: 1. / x[:, :1])
    t_interval = (0., .5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[30, 30],
            trunk_hidden_layer_sizes=[30, 30],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=2e-5),
            epochs=3,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    for i in range(2):
        assert np.all(
            training_loss_history[i + 1].weighted_total_loss.numpy() <
            training_loss_history[i].weighted_total_loss.numpy())

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 6, 11, 3, 1)


def test_pidon_operator_on_spherical_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1., 11.), (0., 2 * np.pi), (.25 * np.pi, .75 * np.pi)],
        [2., np.pi / 5., np.pi / 4],
        CoordinateSystem.SPHERICAL)
    bcs = [
        (DirichletBoundaryCondition(
            lambda x, t: np.ones((len(x), 1)), is_static=True),
         DirichletBoundaryCondition(
             lambda x, t: np.full((len(x), 1), 1. / 11.), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: 1. / x[:, :1])
    t_interval = (0., .5)
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .001, True)

    training_loss_history, test_loss_history = pidon.train(
        cp,
        t_interval,
        training_data_args=DataArgs(
            y_0_functions=[ic.y_0],
            n_domain_points=20,
            n_boundary_points=10,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=20,
            branch_hidden_layer_sizes=[30, 30],
            trunk_hidden_layer_sizes=[30, 30],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=2e-5),
            epochs=3,
            verbose=False
        )
    )

    assert len(training_loss_history) == 3
    for i in range(2):
        assert np.all(
            training_loss_history[i + 1].weighted_total_loss.numpy() <
            training_loss_history[i].weighted_total_loss.numpy())

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 6, 11, 3, 1)


def test_pidon_operator_in_ar_mode_training_with_invalid_t_interval():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., 1.)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression_mode=True)

    with pytest.raises(ValueError):
        pidon.train(
            cp,
            t_interval,
            training_data_args=DataArgs(
                y_0_functions=[ic.y_0],
                n_domain_points=50,
                n_batches=1
            ),
            model_args=ModelArgs(
                latent_output_size=1,
                trunk_hidden_layer_sizes=[50, 50, 50],
            ),
            optimization_args=OptimizationArgs(
                optimizer={'class_name': 'Adam'},
                epochs=100,
                verbose=False
            )
        )


def test_pidon_operator_in_ar_mode_training_with_diff_eq_containing_t_term():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(0, 1)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem([self.symbols.t])

    diff_eq = TestDiffEq()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression_mode=True)

    with pytest.raises(ValueError):
        pidon.train(
            cp,
            (0., .25),
            training_data_args=DataArgs(
                y_0_functions=[ic.y_0],
                n_domain_points=50,
                n_batches=1
            ),
            model_args=ModelArgs(
                latent_output_size=1,
                trunk_hidden_layer_sizes=[50, 50, 50],
            ),
            optimization_args=OptimizationArgs(
                optimizer={'class_name': 'Adam'},
                epochs=100,
                verbose=False
            )
        )


def test_pidon_operator_in_ar_mode_training_with_dynamic_boundary_conditions():
    diff_eq = DiffusionEquation(1, .25)
    mesh = Mesh([(0., .5)], (.05,))
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, t: np.full((len(x), 1), t), is_static=False),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True)),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., .5)
    y_0_functions = [lambda x: np.zeros((len(x), 1))]

    pidon = PIDONOperator(UniformRandomCollocationPointSampler(), .001, True)

    with pytest.raises(ValueError):
        pidon.train(
            cp,
            t_interval,
            training_data_args=DataArgs(
                y_0_functions=y_0_functions,
                n_domain_points=50,
                n_boundary_points=20,
                n_batches=2
            ),
            model_args=ModelArgs(
                latent_output_size=50,
                branch_hidden_layer_sizes=[50, 50],
                trunk_hidden_layer_sizes=[50, 50],
            ),
            optimization_args=OptimizationArgs(
                optimizer={'class_name': 'Adam'},
                epochs=3,
                ic_loss_weight=10.,
                verbose=False
            )
        )


def test_pidon_operator_in_ar_mode_on_ode():
    set_random_seed(0)

    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., 1.)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression_mode=True)

    assert pidon.auto_regression_mode

    pidon.train(
        cp,
        (0., .25),
        training_data_args=DataArgs(
            y_0_functions=[
                lambda _, _y_0=y_0: np.array([_y_0])
                for y_0 in np.linspace(.5, 2., 10)
            ],
            n_domain_points=50,
            n_batches=1
        ),
        model_args=ModelArgs(
            latent_output_size=1,
            trunk_hidden_layer_sizes=[50, 50, 50],
        ),
        optimization_args=OptimizationArgs(
            optimizer={
                'class_name': 'Adam',
                'config': {
                    'learning_rate': optimizers.schedules.ExponentialDecay(
                        1e-2, decay_steps=25, decay_rate=.95)
                }
            },
            epochs=5,
            verbose=False
        )
    )

    sol = pidon.solve(ivp)
    assert np.allclose(sol.t_coordinates, [.25, .5, .75, 1.])
    assert sol.discrete_y().shape == (4, 1)


def test_pidon_operator_in_ar_mode_on_pde():
    set_random_seed(0)

    diff_eq = WaveEquation(1)
    mesh = Mesh([(0., 1.)], (.2,))
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 2)), is_static=True)),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 1.)
    ic = BetaInitialCondition(cp, [(3.5, 3.5), (3.5, 3.5)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    training_y_0_functions = [
        BetaInitialCondition(cp, [(p, p), (p, p)]).y_0
        for p in [2., 3., 4., 5.]
    ]
    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, False, auto_regression_mode=True)

    assert pidon.auto_regression_mode

    pidon.train(
        cp,
        (0., .25),
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=50,
            n_boundary_points=20,
            n_batches=2
        ),
        model_args=ModelArgs(
            latent_output_size=50,
            branch_hidden_layer_sizes=[50, 50],
            trunk_hidden_layer_sizes=[50, 50],
        ),
        optimization_args=OptimizationArgs(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            epochs=2,
            ic_loss_weight=10.,
            verbose=False
        )
    )

    sol = pidon.solve(ivp)
    assert np.allclose(sol.t_coordinates, [.25, .5, .75, 1.])
    assert sol.discrete_y().shape == (4, 5, 2)
