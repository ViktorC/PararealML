from typing import Callable, Tuple, Optional

import numpy as np


class BoundaryCondition:
    """
    A base class for boundary conditions.
    """

    def has_y_condition(self) -> bool:
        """
        Returns whether the boundary conditions restrict the value of y.
        """
        raise NotImplementedError

    def has_d_y_condition(self) -> bool:
        """
        Returns whether the boundary conditions restrict the value of the
        derivative of y with respect to the normal vector of the boundary.
        """
        raise NotImplementedError

    def y_condition(self, x: Tuple[float, ...]) -> Optional[np.ndarray]:
        """
        Returns the value of y at the coordinates along the boundary specified
        by x. To avoid imposing a condition on elements of y, the corresponding
        elements of the returned array may be NaNs.

        :param x: the coordinates in the hyperplane of the boundary
        :return: the value of y(x)
        """
        pass

    def d_y_condition(self, x: Tuple[float, ...]) -> Optional[np.ndarray]:
        """
        Returns the value of the derivative of y at the coordinates along the
        boundary specified by x with respect to the normal vector to the
        boundary passing through the same point. To avoid imposing a condition
        on elements of the spatial derivative of elements of y, the
        corresponding elements of the returned array may be NaNs.

        :param x: the coordinates in the hyperplane of the boundary
        :return: the constrained value of dy(x) / dn
        """
        pass


class DirichletCondition(BoundaryCondition):
    """
    Dirichlet boundary conditions that restrict the values of y along the
    boundary.
    """

    def __init__(self, y_condition: Callable[[Tuple[float, ...]], np.ndarray]):
        """
        :param y_condition: the function that determines the value of y at the
        coordinates along the boundary specified by x
        """
        self._y_condition = y_condition

    def has_y_condition(self) -> bool:
        return True

    def has_d_y_condition(self) -> bool:
        return False

    def y_condition(self, x: Tuple[float, ...]) -> np.ndarray:
        return self._y_condition(x)


class NeumannCondition(BoundaryCondition):
    """
    Neumann boundary conditions that restrict the values of the derivative of y
    with respect to the normal of the boundary.
    """

    def __init__(
            self,
            d_y_condition: Callable[[Tuple[float, ...]], np.ndarray]):
        """
        :param d_y_condition: the function that determines the value of the
        derivative of y at the coordinates along the boundary specified by x
        with respect to the normal vector to the boundary passing through the
        same point
        """
        self._d_y_condition = d_y_condition

    def has_y_condition(self) -> bool:
        return False

    def has_d_y_condition(self) -> bool:
        return True

    def d_y_condition(self, x: Tuple[float, ...]) -> np.ndarray:
        return self._d_y_condition(x)


class CauchyCondition(BoundaryCondition):
    """
    A combination of Dirichlet and Neumann boundary conditions.
    """

    def __init__(
            self,
            y_condition: Callable[[Tuple[float, ...]], np.ndarray],
            d_y_condition: Callable[[Tuple[float, ...]], np.ndarray]):
        """
        :param y_condition: the function that determines the value of y at the
        coordinates along the boundary specified by x
        :param d_y_condition: the function that determines the value of the
        derivative of y at the coordinates along the boundary specified by x
        with respect to the normal vector to the boundary passing through the
        same point
        """
        self._y_condition = y_condition
        self._d_y_condition = d_y_condition

    def has_y_condition(self) -> bool:
        return True

    def has_d_y_condition(self) -> bool:
        return True

    def y_condition(self, x: Tuple[float, ...]) -> np.ndarray:
        return self._y_condition(x)

    def d_y_condition(self, x: Tuple[float, ...]) -> np.ndarray:
        return self._d_y_condition(x)
