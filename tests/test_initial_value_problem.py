import numpy as np
import pytest

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import PopulationGrowthEquation
from pararealml.initial_condition import ContinuousInitialCondition
from pararealml.initial_value_problem import InitialValueProblem


def test_initial_value_problem_with_invalid_time_interval():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([0.]))

    with pytest.raises(ValueError):
        InitialValueProblem(cp, (3., 2.), ic)


def test_initial_value_problem_without_exact_solution():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([0.]))
    ivp = InitialValueProblem(cp, (0., 2.), ic)

    assert not ivp.has_exact_solution

    with pytest.raises(RuntimeError):
        ivp.exact_y(2.)


def test_initial_value_problem():
    y_0 = 100
    r = .5
    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))
    ivp = InitialValueProblem(
        cp,
        (0., 2.),
        ic,
        lambda _ivp, t, x: np.array([y_0 * np.e ** (r * t)])
    )

    assert ivp.has_exact_solution
    assert ivp.constrained_problem == cp
    assert ivp.t_interval == (0., 2.)
    assert ivp.initial_condition == ic
    assert np.allclose(ivp.exact_y(0.), [y_0])
