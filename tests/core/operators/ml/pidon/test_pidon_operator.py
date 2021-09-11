import numpy as np
from tensorflow import optimizers

from pararealml.core.boundary_condition import NeumannBoundaryCondition
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import PopulationGrowthEquation, \
    LotkaVolterraEquation, DiffusionEquation
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
            domain_batch_size=5
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
    assert training_loss_history[-1].weighted_total_loss.numpy() < 5e-6

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

    assert np.mean(np.abs(analytic_y - solution.discrete_y())) < 5e-4
    assert np.max(np.abs(analytic_y - solution.discrete_y())) < 1e-3


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
            domain_batch_size=100
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=20,
            domain_batch_size=60
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
            domain_batch_size=100,
            boundary_batch_size=40,
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=25,
            n_boundary_points=10,
            domain_batch_size=75,
            boundary_batch_size=30,
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
