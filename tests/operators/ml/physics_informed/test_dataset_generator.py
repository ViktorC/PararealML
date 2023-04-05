import numpy as np
import pytest

from pararealml.boundary_condition import (
    CauchyBoundaryCondition,
    DirichletBoundaryCondition,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    CahnHilliardEquation,
    DiffusionEquation,
    LotkaVolterraEquation,
    PopulationGrowthEquation,
)
from pararealml.mesh import Mesh
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    UniformRandomCollocationPointSampler,
)
from pararealml.operators.ml.physics_informed.dataset_generator import (
    DatasetGenerator,
)


def test_dataset_generator_on_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0.0, 100.0)
    y_0_functions = [
        lambda _: np.array([10.0, 20.0]),
        lambda _: np.array([15.0, 15.0]),
        lambda _: np.array([20.0, 10.0]),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 200

    dataset_generator = DatasetGenerator(
        cp, t_interval, y_0_functions, sampler, n_points
    )

    assert np.array_equal(
        dataset_generator.initial_value_data,
        np.array([[10.0, 20.0], [15.0, 15.0], [20.0, 10.0]]),
    )
    assert dataset_generator.domain_collocation_data.shape == (200, 1)
    assert np.allclose(dataset_generator.initial_collocation_data, [[0.0]])
    assert dataset_generator.boundary_collocation_data is None


def test_dataset_generator_on_pde():
    diff_eq = CahnHilliardEquation(2)
    mesh = Mesh([(0.0, 5.0), (0.0, 2.0)], [0.5, 0.25])
    bcs = [
        (
            CauchyBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0)),
                vectorize_bc_function(lambda x, t: (1.0, 1.0)),
                is_static=True,
            ),
            CauchyBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0)),
                vectorize_bc_function(lambda x, t: (1.0, 1.0)),
                is_static=True,
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 10.0)
    y_0_functions = [
        lambda x: np.stack(
            [
                x[:, 0] ** 2 - 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2,
                x[:, 1] ** 0.5,
            ],
            axis=-1,
        ),
        lambda x: np.stack(
            [x[:, 0] ** 3 - x[:, 1] ** 3, x[:, 0] ** 0.5], axis=-1
        ),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    dataset_generator = DatasetGenerator(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points,
    )

    assert dataset_generator.initial_value_data.shape == (2, 80 * 2)
    assert dataset_generator.domain_collocation_data.shape == (200, 1 + 2)
    assert dataset_generator.initial_collocation_data.shape == (80, 1 + 2)
    assert dataset_generator.boundary_collocation_data.shape == (
        50,
        1 + 2 + 2 + 2 + 1,
    )

    assert np.all(dataset_generator.boundary_collocation_data[:, 3:5] == 0.0)
    assert np.all(dataset_generator.boundary_collocation_data[:, 5:7] == 1.0)


def test_dataset_generator_generate_raises_error_if_n_batches_not_divisor():
    cp = ConstrainedProblem(PopulationGrowthEquation())
    sampler = UniformRandomCollocationPointSampler()
    dataset_generator = DatasetGenerator(
        cp, (0.0, 5.0), [lambda _: np.array([5.0])], sampler, 100
    )

    with pytest.raises(ValueError):
        dataset_generator.generate(2)


def test_dataset_generator_generate_on_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0.0, 40.0)
    y_0_functions = [
        lambda _: np.array([10.0, 20.0]),
        lambda _: np.array([15.0, 15.0]),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 5

    dataset_generator = DatasetGenerator(
        cp, t_interval, y_0_functions, sampler, n_points
    )
    dataset = dataset_generator.generate(5, n_ic_repeats=5)

    assert len(dataset) == 5

    shuffled_batches = list(dataset.as_numpy_iterator())
    assert len(shuffled_batches) == 5
    for batch in shuffled_batches:
        assert batch["domain"]["u"].shape == (2, 2)
        assert batch["domain"]["t"].shape == (2, 1)
        assert "x" not in batch["domain"]

        assert batch["initial"]["u"].shape == (2, 2)
        assert batch["initial"]["t"].shape == (2, 1)
        assert "x" not in batch["initial"]
        assert batch["initial"]["y"].shape == (2, 2)

        assert "boundary" not in batch

    batches = list(
        dataset_generator.generate(
            5, n_ic_repeats=5, shuffle=False
        ).as_numpy_iterator()
    )
    assert len(batches) == 5

    assert np.allclose(batches[0]["domain"]["u"], [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(batches[1]["domain"]["u"], [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(batches[2]["domain"]["u"], [[10.0, 20.0], [15.0, 15.0]])
    assert np.allclose(batches[3]["domain"]["u"], [[15.0, 15.0], [15.0, 15.0]])
    assert np.allclose(batches[4]["domain"]["u"], [[15.0, 15.0], [15.0, 15.0]])


def test_dataset_generator_generate_on_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0.0, 5.0), (0.0, 5.0)], [0.1, 0.1])
    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0,)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0,)), is_static=True
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 5.0)
    y_0_functions = [
        lambda x: x[:, :1] ** 2 - x[:, 1:] ** 2,
        lambda x: x[:, :1] * x[:, 1:] / (x[:, :1] ** 2 + x[:, 1:] ** 2),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    dataset_generator = DatasetGenerator(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points,
    )
    dataset = dataset_generator.generate(2)

    assert len(dataset) == 2

    batches = list(dataset.as_numpy_iterator())
    assert len(batches) == 2
    for batch in batches:
        assert batch["domain"]["u"].shape == (200, 2500)
        assert batch["domain"]["t"].shape == (200, 1)
        assert batch["domain"]["x"].shape == (200, 2)

        assert batch["initial"]["u"].shape == (2500, 2500)
        assert batch["initial"]["t"].shape == (2500, 1)
        assert batch["initial"]["x"].shape == (2500, 2)
        assert batch["initial"]["y"].shape == (2500, 1)

        assert batch["boundary"]["u"].shape == (50, 2500)
        assert batch["boundary"]["t"].shape == (50, 1)
        assert batch["boundary"]["x"].shape == (50, 2)
        assert batch["boundary"]["y"].shape == (50, 1)
        assert batch["boundary"]["d_y_over_d_n"].shape == (50, 1)
        assert batch["boundary"]["axes"].shape == (50,)

        assert np.all(batch["boundary"]["y"] == 0.0)
        assert np.isnan(batch["boundary"]["d_y_over_d_n"]).all()
