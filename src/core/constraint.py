from typing import Union, Optional, Sequence

import numpy as np


class Constraint:
    """
    A representation of constraints on the values of an array.
    """

    def __init__(self, value: np.ndarray, mask: np.ndarray):
        """
        :param value: the constraint values
        :param mask: the mask denoting which elements of the array are to be
            constrained to the provided values
        """
        assert value.size == mask.sum()
        self._value = np.copy(value)
        self._mask = np.copy(mask)

    @property
    def value(self) -> np.ndarray:
        """
        Returns the constraint values.
        """
        return np.copy(self._value)

    @property
    def mask(self) -> np.ndarray:
        """
        Returns the mask denoting the elements of the array that are to be
        constrained.
        """
        return np.copy(self._mask)

    def apply(self, array: np.ndarray) -> np.ndarray:
        """
        It applies the constraints to the provided array in-place and returns
        the constrained array.

        :param array: the array to constrain
        :return: the constrained array
        """
        assert array.shape[-self._mask.ndim:] == self._mask.shape

        array[..., self._mask] = self._value
        return array

    def multiply_and_add(
            self,
            addend: np.ndarray,
            multiplier: Union[float, np.ndarray],
            result: np.ndarray
    ) -> np.ndarray:
        """
        It constrains the result array in-place to the sum of the constraint
        values multiplied by the provided multiplier and the corresponding
        elements of the provided addend. It also returns the constrained
        result array.

        :param addend: the array whose values selected by the mask are to be
            added to the multiplied constraint values
        :param multiplier: the factor by which the constraint values are to be
            multiplied
        :param result: the array to constrain by the result of the operation
        :return: the constrained result array
        """
        assert addend.shape == result.shape
        assert addend.shape[-self._mask.ndim:] == self._mask.shape
        assert isinstance(multiplier, float) \
            or multiplier.shape == self._value.shape

        result[..., self._mask] = addend[..., self._mask] + \
            multiplier * self._value
        return result


def apply_constraints_along_last_axis(
        constraints: Optional[Sequence[Optional[Constraint]]],
        array: np.ndarray) -> np.ndarray:
    """
    Applies the provided constraints to the array in-place and returns
    the constrained array.

    :param constraints: the constraints on the values of the array
    :param array: the array to which the constraints are to be applied
    :return: the constrained array
    """
    if constraints is not None:
        assert array.ndim > 1
        assert len(constraints) == array.shape[-1]

        for i, constraint in enumerate(constraints):
            if constraint is not None:
                constraint.apply(array[..., i])

    return array
