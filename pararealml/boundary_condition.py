from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np

VectorizedBoundaryConditionFunction = Callable[
    [np.ndarray, Optional[float]], np.ndarray
]


class BoundaryCondition(ABC):
    """
    A base class for boundary conditions.
    """

    def __init__(
        self, has_y_condition: bool, has_d_y_condition: bool, is_static: bool
    ):
        """
        :param has_y_condition: whether the boundary conditions restrict the
            value of y
        :param has_d_y_condition: whether the boundary conditions restrict the
            value of the derivative of y with respect to the normal vector of
            the boundary
        :param is_static: whether the boundary condition is time independent
        """
        self._has_y_condition = has_y_condition
        self._has_d_y_condition = has_d_y_condition
        self._is_static = is_static

    @property
    def has_y_condition(self) -> bool:
        """
        Whether the boundary conditions restrict the value of y.
        """
        return self._has_y_condition

    @property
    def has_d_y_condition(self) -> bool:
        """
        Whether the boundary conditions restrict the value of the derivative of
        y with respect to the normal vector of the boundary.
        """
        return self._has_d_y_condition

    @property
    def is_static(self) -> bool:
        """
        Whether the boundary condition is time independent.
        """
        return self._is_static

    @abstractmethod
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        """
        Returns the value of y at the coordinates along the boundary specified
        by x. To avoid imposing a condition on elements of y, the corresponding
        elements of the returned array may be NaNs.

        :param x: a 2D array (n, x_dimension) of the boundary coordinates
        :param t: the time value; if the condition is static, it may be None
        :return: a 2D array (n, y_dimension) of the constrained value of y at
            the boundary points
        """

    @abstractmethod
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        """
        Returns the value of the derivative of y at the coordinates along the
        boundary specified by x with respect to the normal vector to the
        boundary passing through the same point. To avoid imposing a condition
        on elements of the spatial derivative of elements of y, the
        corresponding elements of the returned array may be NaNs.

        :param x: a 2D array (n, x_dimension) of the boundary coordinates
        :param t: the time value; if the condition is static, it may be None
        :return: a 2D array (n, y_dimension) of the constrained value of the
            derivative of y with respect to the normal vector to the boundary
            at the points defined by x
        """


class DirichletBoundaryCondition(BoundaryCondition):
    """
    Dirichlet boundary conditions that restrict the values of y along the
    boundary.
    """

    def __init__(
        self,
        y_condition: VectorizedBoundaryConditionFunction,
        is_static: bool = False,
    ):
        """
        :param y_condition: the function that determines the value of y at the
        coordinates along the boundary specified by x
        :param is_static: whether the boundary condition is time independent
        """
        self._y_condition = y_condition

        super(DirichletBoundaryCondition, self).__init__(
            True, False, is_static
        )

    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._y_condition(x, t)

    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise RuntimeError(
            "Dirichlet conditions do not constrain the normal derivative of y"
        )


class NeumannBoundaryCondition(BoundaryCondition):
    """
    Neumann boundary conditions that restrict the values of the derivative of y
    with respect to the normal of the boundary.
    """

    def __init__(
        self,
        d_y_condition: VectorizedBoundaryConditionFunction,
        is_static: bool = False,
    ):
        """
        :param d_y_condition: the function that determines the value of the
            derivative of y at the coordinates along the boundary specified by
            x with respect to the normal vector to the boundary
        :param is_static: whether the boundary condition is time independent
        """
        self._d_y_condition = d_y_condition

        super(NeumannBoundaryCondition, self).__init__(False, True, is_static)

    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise RuntimeError("Neumann conditions do not constrain y")

    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._d_y_condition(x, t)


class CauchyBoundaryCondition(BoundaryCondition):
    """
    A combination of Dirichlet and Neumann boundary conditions.
    """

    def __init__(
        self,
        y_condition: VectorizedBoundaryConditionFunction,
        d_y_condition: VectorizedBoundaryConditionFunction,
        is_static: bool = False,
    ):
        """
        :param y_condition: the function that determines the value of y at the
            coordinates along the boundary specified by x
        :param d_y_condition: the function that determines the value of the
            derivative of y at the coordinates along the boundary specified by
            x with respect to the normal vector to the boundary
        :param is_static: whether the boundary condition is time independent
        """
        self._y_condition = y_condition
        self._d_y_condition = d_y_condition

        super(CauchyBoundaryCondition, self).__init__(True, True, is_static)

    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._y_condition(x, t)

    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._d_y_condition(x, t)


class ConstantBoundaryCondition(BoundaryCondition):
    """
    A set of constant, space and time independent boundary conditions.
    """

    def __init__(
        self,
        constant_y_conditions: Optional[Sequence[Optional[float]]],
        constant_d_y_conditions: Optional[Sequence[Optional[float]]],
    ):
        """
        :param constant_y_conditions: a sequence of scalars each denoting the
            boundary conditions on the corresponding element of y
        :param constant_d_y_conditions: a sequence of scalars each denoting the
            boundary conditions on the corresponding element of the normal
            derivative of y
        """
        if constant_y_conditions is None and constant_d_y_conditions is None:
            raise ValueError(
                "at least one type of constant conditions must not be None"
            )

        self._constant_y_conditions = constant_y_conditions
        self._constant_d_y_conditions = constant_d_y_conditions

        super(ConstantBoundaryCondition, self).__init__(
            constant_y_conditions is not None,
            constant_d_y_conditions is not None,
            True,
        )

    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        if not self._constant_y_conditions:
            raise RuntimeError("no boundary conditions defined on y")

        return np.hstack(
            [
                np.full((len(x), 1), y_condition)
                for y_condition in self._constant_y_conditions
            ]
        )

    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        if not self._constant_d_y_conditions:
            raise RuntimeError(
                "no boundary conditions defined on the normal derivative of y"
            )

        return np.hstack(
            [
                np.full((len(x), 1), d_y_condition)
                for d_y_condition in self._constant_d_y_conditions
            ]
        )


class ConstantValueBoundaryCondition(ConstantBoundaryCondition):
    """
    A set of constant, space and time independent boundary conditions on the
    value of the solution.
    """

    def __init__(self, constant_y_conditions: Sequence[Optional[float]]):
        """
        :param constant_y_conditions: a sequence of scalars each denoting the
            boundary conditions on the corresponding element of y
        """
        super(ConstantValueBoundaryCondition, self).__init__(
            constant_y_conditions, None
        )


class ConstantFluxBoundaryCondition(ConstantBoundaryCondition):
    """
    A set of constant, space and time independent boundary conditions on the
    normal derivative of the solution.
    """

    def __init__(self, constant_d_y_conditions: Sequence[Optional[float]]):
        """
        :param constant_d_y_conditions: a sequence of scalars each denoting the
            boundary conditions on the corresponding element of the normal
            derivative of y
        """
        super(ConstantFluxBoundaryCondition, self).__init__(
            None, constant_d_y_conditions
        )


def vectorize_bc_function(
    bc_function: Callable[
        [Sequence[float], Optional[float]], Sequence[Optional[float]]
    ]
) -> VectorizedBoundaryConditionFunction:
    """
    Vectorizes a boundary condition function that operates on a single
    coordinate sequence so that it can operate on an array of coordinate
    sequences.

    The implementation of the vectorized function is nothing more than a for
    loop over the rows of coordinate sequences in the x argument.

    :param bc_function: the non-vectorized boundary condition function
    :return: the vectorized boundary condition function
    """

    def vectorized_bc_function(
        x: np.ndarray, t: Optional[float]
    ) -> np.ndarray:
        values = []
        for i in range(len(x)):
            values.append(bc_function(x[i], t))
        return np.array(values, dtype=float)

    return vectorized_bc_function
