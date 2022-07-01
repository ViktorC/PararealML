from typing import Optional, Sequence, Union

import numpy as np


class Constraint:
    """
    A representation of constraints on the values of an array.
    """

    def __init__(self, values: np.ndarray, mask: np.ndarray):
        """
        :param values: the constraint values
        :param mask: the mask denoting which elements of the array are to be
            constrained to the provided values
        """
        if values.size != mask.sum():
            raise ValueError(
                f"number of values ({values.size}) must match number "
                f"of True elements in mask ({mask.sum()})"
            )

        self._values = np.copy(values)
        self._mask = np.copy(mask)

        self._values.setflags(write=False)
        self._mask.setflags(write=False)

    @property
    def values(self) -> np.ndarray:
        """
        The constraint values.
        """
        return self._values

    @property
    def mask(self) -> np.ndarray:
        """
        The mask denoting the elements of the array that are to be constrained.
        """
        return self._mask

    def apply(self, array: np.ndarray) -> np.ndarray:
        """
        It applies the constraints to the provided array in-place and returns
        the constrained array.

        :param array: the array to constrain
        :return: the constrained array
        """
        if array.shape[-self._mask.ndim :] != self._mask.shape:
            raise ValueError(
                f"input shape {array.shape} incompatible with mask shape "
                f"{self._mask.shape}"
            )

        array[..., self._mask] = self._values
        return array

    def multiply_and_add(
        self,
        addend: np.ndarray,
        multiplier: Union[float, np.ndarray],
        result: np.ndarray,
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
        if addend.shape != result.shape:
            raise ValueError(
                f"addend shape {addend.shape} must match result shape "
                f"{result.shape}"
            )
        if result.shape[-self._mask.ndim :] != self._mask.shape:
            raise ValueError(
                f"result shape {result.shape} incompatible with mask shape "
                f"{self._mask.shape}"
            )
        if (
            not isinstance(multiplier, float)
            and multiplier.shape != self._values.shape
        ):
            raise ValueError(
                f"multiplier shape {multiplier.shape} must match values shape "
                f"{self._values.shape}"
            )

        result[..., self._mask] = (
            addend[..., self._mask] + multiplier * self._values
        )
        return result


def apply_constraints_along_last_axis(
    constraints: Optional[Union[Sequence[Optional[Constraint]], np.ndarray]],
    array: np.ndarray,
) -> np.ndarray:
    """
    Applies the provided constraints to the array in-place and returns
    the constrained array.

    :param constraints: the constraints on the values of the array
    :param array: the array to which the constraints are to be applied
    :return: the constrained array
    """
    if constraints is not None:
        if array.ndim <= 1:
            raise ValueError(
                f"input dimensions ({array.ndim}) must be at least 2"
            )
        if len(constraints) != array.shape[-1]:
            raise ValueError(
                f"number of constraints ({len(constraints)}) must match the "
                f"size of the input array's last axis ({array.shape[-1]})"
            )

        for i, constraint in enumerate(constraints):
            if constraint is not None:
                constraint.apply(array[..., i : i + 1])

    return array
