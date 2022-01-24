import numpy as np
import pytest

from pararealml.constraint import Constraint, apply_constraints_along_last_axis


def create_3_by_3_test_constraint() -> Constraint:
    array = np.full((3, 3), np.nan)
    array[0, 0] = 1.
    array[1, 1] = 2.
    array[2, 2] = 3.
    mask = ~np.isnan(array)
    value = array[mask]
    return Constraint(value, mask)


def create_4_by_1_test_constraint() -> Constraint:
    array = np.full((4, 1), np.nan)
    array[0, 0] = 1.
    array[2, 0] = 3.
    mask = ~np.isnan(array)
    value = array[mask]
    return Constraint(value, mask)


def test_apply_with_wrong_array_shape():
    constraint = create_3_by_3_test_constraint()

    with pytest.raises(ValueError):
        constraint.apply(np.zeros((2, 3)))


def test_apply():
    constraint = create_3_by_3_test_constraint()

    result = np.zeros((1, 3, 3))
    expected_result = [[
        [1., 0., 0.],
        [0., 2., 0.],
        [0., 0., 3.],
    ]]
    constraint.apply(result)
    assert np.array_equal(result, expected_result)


def test_multiply_and_add_with_wrong_result_shape():
    constraint = create_3_by_3_test_constraint()

    result = np.zeros((2, 3))
    addend = np.zeros_like(result)
    multiplier = 1.

    with pytest.raises(ValueError):
        constraint.multiply_and_add(addend, multiplier, result)


def test_multiply_and_add_with_mismatched_result_and_addend_shape():
    constraint = create_3_by_3_test_constraint()

    result = np.zeros((3, 3))
    addend = np.zeros((3, 4))
    multiplier = 1.

    with pytest.raises(ValueError):
        constraint.multiply_and_add(addend, multiplier, result)


def test_multiply_and_add_with_wrong_multiplier_shape():
    constraint = create_3_by_3_test_constraint()

    result = np.zeros((3, 3))
    addend = np.zeros_like(result)
    multiplier = np.ones(4)

    with pytest.raises(ValueError):
        constraint.multiply_and_add(addend, multiplier, result)


def test_multiply_and_add():
    constraint = create_3_by_3_test_constraint()

    result = np.zeros((3, 3))
    addend = np.ones_like(result)
    multiplier = 3.
    expected_result = [
        [4., 0., 0.],
        [0., 7., 0.],
        [0., 0., 10.],
    ]
    constraint.multiply_and_add(addend, multiplier, result)
    assert np.array_equal(result, expected_result)


def test_apply_constraints_along_last_axis_with_one_dimensional_array():
    constraints = [
        create_4_by_1_test_constraint(), create_4_by_1_test_constraint()
    ]

    array = np.zeros(1)

    with pytest.raises(ValueError):
        apply_constraints_along_last_axis(constraints, array)


def test_apply_constraints_along_last_axis_with_wrong_last_array_axis_size():
    constraints = [
        create_4_by_1_test_constraint(), create_4_by_1_test_constraint()
    ]

    array = np.zeros((3, 3, 1))

    with pytest.raises(ValueError):
        apply_constraints_along_last_axis(constraints, array)


def test_apply_constraints_along_last_axis():
    constraints = [
        create_4_by_1_test_constraint(), create_4_by_1_test_constraint()
    ]

    array = np.zeros((1, 4, 2))
    expected_array = [[
        [1., 1.],
        [0., 0.],
        [3., 3.],
        [0., 0.]
    ]]
    apply_constraints_along_last_axis(constraints, array)
    assert np.array_equal(array, expected_array)
