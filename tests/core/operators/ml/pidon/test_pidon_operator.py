import numpy as np
import pytest
from tensorflow import optimizers

from pararealml.core.boundary_condition import NeumannBoundaryCondition
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import PopulationGrowthEquation, \
    LotkaVolterraEquation, DiffusionEquation, WaveEquation
from pararealml.core.initial_condition import ContinuousInitialCondition, \
    GaussianInitialCondition
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.mesh import Mesh
from pararealml.core.operators.ml.pidon.collocation_point_sampler import \
    UniformRandomCollocationPointSampler
from pararealml.core.operators.ml.pidon.pidon_operator import PIDONOperator, \
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
            optimizer={
                'class_name': 'Adam',
                'config': {
                    'learning_rate': optimizers.schedules.ExponentialDecay(
                        1e-2, decay_steps=25, decay_rate=.95)
                }
            },
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
    assert training_loss_history[-1].weighted_total_loss.numpy() < 1e-5

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


def test_pidon_operator_on_pde():
    set_random_seed(0)

    diff_eq = DiffusionEquation(1, .25)
    mesh = Mesh([(0., .5)], (.05,))
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 1)), is_static=True)),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., .5)

    ic_mean = .25
    training_y_0_functions = [
        GaussianInitialCondition(
            cp,
            [(
                np.array([ic_mean]),
                np.array([[sd]])
            )]
        ).y_0 for sd in [.1, .2, .3, .4]
    ]
    test_y_0_functions = [
        GaussianInitialCondition(
            cp,
            [(
                np.array([ic_mean]),
                np.array([[sd]])
            )]
        ).y_0 for sd in [.15, .25, .35]
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

    ic = GaussianInitialCondition(
        cp,
        [(np.array([ic_mean]), np.array([[.25]]))]
    )
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = pidon.solve(ivp)
    assert solution.d_t == .001
    assert solution.discrete_y().shape == (500, 11, 1)


def test_pidon_operator_in_auto_regression_mode_with_invalid_t_interval():
    set_random_seed(0)

    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., 1.)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression=True)

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
                optimizer={
                    'class_name': 'Adam',
                    'config': {
                        'learning_rate': optimizers.schedules.ExponentialDecay(
                            1e-2, decay_steps=25, decay_rate=.95)
                    }
                },
                epochs=100,
                verbose=False
            )
        )


def test_pidon_operator_on_ode_in_auto_regression_mode():
    set_random_seed(0)

    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    t_interval = (0., 1.)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression=True)
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


def test_pidon_operator_on_pde_in_auto_regression_mode():
    set_random_seed(0)

    diff_eq = WaveEquation(1)
    mesh = Mesh([(0., .5)], (.1,))
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 2)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: np.zeros((len(x), 2)), is_static=True)),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 1.)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([.5]), np.array([[.25]]))] * 2
    )
    ivp = InitialValueProblem(cp, t_interval, ic)

    training_y_0_functions = [
        GaussianInitialCondition(
            cp,
            [(
                np.array([.5]),
                np.array([[sd]])
            )] * 2
        ).y_0 for sd in [.1, .2, .3, .4]
    ]
    sampler = UniformRandomCollocationPointSampler()
    pidon = PIDONOperator(sampler, .25, True, auto_regression=True)
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
            optimizer={
                'class_name': 'Adam',
                'config': {
                    'learning_rate': 1e-4
                }
            },
            epochs=2,
            ic_loss_weight=10.,
            verbose=False
        )
    )

    sol = pidon.solve(ivp)
    assert np.allclose(sol.t_coordinates, [.25, .5, .75, 1.])
    assert sol.discrete_y().shape == (4, 6, 2)
