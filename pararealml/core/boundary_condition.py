from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

BoundaryConditionFunction = Callable[
    [Sequence[float], Optional[float]],
    Sequence[Optional[float]]
]


class BoundaryCondition(ABC):
    """
    A base class for boundary conditions.
    """

    @property
    @abstractmethod
    def is_static(self) -> bool:
        """
        Whether the boundary condition is time independent.
        """

    @property
    @abstractmethod
    def has_y_condition(self) -> bool:
        """
        Whether the boundary conditions restrict the value of y.
        """

    @property
    @abstractmethod
    def has_d_y_condition(self) -> bool:
        """
        Whether the boundary conditions restrict the value of the derivative of
        y with respect to the normal vector of the boundary.
        """

    @abstractmethod
    def y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        """
        Returns the value of y at the coordinates along the boundary specified
        by x. To avoid imposing a condition on elements of y, the corresponding
        elements of the returned sequence may be None.

        :param x: the boundary coordinates
        :param t: the time value; if the condition is static, it may be None
        :return: the value of y(x, t)
        """

    @abstractmethod
    def d_y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        """
        Returns the value of the derivative of y at the coordinates along the
        boundary specified by x with respect to the normal vector to the
        boundary passing through the same point. To avoid imposing a condition
        on elements of the spatial derivative of elements of y, the
        corresponding elements of the returned sequence may be None.

        :param x: the boundary coordinates
        :param t: the time value; if the condition is static, it may be None
        :return: the constrained value of dy(x, t) / dn
        """


class DirichletBoundaryCondition(BoundaryCondition):
    """
    Dirichlet boundary conditions that restrict the values of y along the
    boundary.
    """

    def __init__(
            self,
            y_condition: BoundaryConditionFunction,
            is_static: bool = False):
        """
        :param y_condition: the function that determines the value of y at the
        coordinates along the boundary specified by x
        :param is_static: whether the boundary condition is time independent
        """
        self._y_condition = y_condition
        self._is_static = is_static

    @property
    def is_static(self) -> bool:
        return self._is_static

    @property
    def has_y_condition(self) -> bool:
        return True

    @property
    def has_d_y_condition(self) -> bool:
        return False

    def y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        return self._y_condition(x, t)

    def d_y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        pass


class NeumannBoundaryCondition(BoundaryCondition):
    """
    Neumann boundary conditions that restrict the values of the derivative of y
    with respect to the normal of the boundary.
    """

    def __init__(
            self,
            d_y_condition: BoundaryConditionFunction,
            is_static: bool = False):
        """
        :param d_y_condition: the function that determines the value of the
            derivative of y at the coordinates along the boundary specified by
            x with respect to the normal vector to the boundary passing through
            the same point
        :param is_static: whether the boundary condition is time independent
        """
        self._d_y_condition = d_y_condition
        self._is_static = is_static

    @property
    def is_static(self) -> bool:
        return self._is_static

    @property
    def has_y_condition(self) -> bool:
        return False

    @property
    def has_d_y_condition(self) -> bool:
        return True

    def y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        pass

    def d_y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        return self._d_y_condition(x, t)


class CauchyBoundaryCondition(BoundaryCondition):
    """
    A combination of Dirichlet and Neumann boundary conditions.
    """

    def __init__(
            self,
            y_condition: BoundaryConditionFunction,
            d_y_condition: BoundaryConditionFunction,
            is_static: bool = False):
        """
        :param y_condition: the function that determines the value of y at the
            coordinates along the boundary specified by x
        :param d_y_condition: the function that determines the value of the
            derivative of y at the coordinates along the boundary specified by
            x with respect to the normal vector to the boundary passing through
            the same point
        :param is_static: whether the boundary condition is time independent
        """
        self._y_condition = y_condition
        self._d_y_condition = d_y_condition
        self._is_static = is_static

    @property
    def is_static(self) -> bool:
        return self._is_static

    @property
    def has_y_condition(self) -> bool:
        return True

    @property
    def has_d_y_condition(self) -> bool:
        return True

    def y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        return self._y_condition(x, t)

    def d_y_condition(
            self,
            x: Sequence[float],
            t: Optional[float]
    ) -> Optional[Sequence[Optional[float]]]:
        return self._d_y_condition(x, t)
