import numpy as np
import pytest

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import PopulationGrowthEquation
from pararealml.core.initial_condition import ContinuousInitialCondition
from pararealml.core.initial_value_problem import InitialValueProblem


def test_initial_value_problem_with_invalid_time_interval():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: [0.])

    with pytest.raises(ValueError):
        InitialValueProblem(cp, (3., 2.), ic)


def test_initial_value_problem():
    y_0 = 100
    r = .5
    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: [y_0])
    ivp = InitialValueProblem(
        cp,
        (0., 2.),
        ic,
        lambda _ivp, t, x: [y_0 * np.e ** (r * t)])

    assert ivp.has_exact_solution
    assert ivp.constrained_problem == cp
    assert ivp.t_interval == (0., 2.)
    assert ivp.initial_condition == ic
    assert np.isclose(ivp.exact_y(0.), [y_0]).all()
