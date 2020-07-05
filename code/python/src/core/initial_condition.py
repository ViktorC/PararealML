from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, Optional, Callable

import numpy as np

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.constraint import apply_constraints_along_last_axis


class InitialCondition(ABC):
    """
    A base class for initial conditions.
    """

    @abstractmethod
    def is_well_defined(self) -> bool:
        """
        Returns whether the initial conditions are well defined or approximated
        over a mesh. In case of the latter, the y_0 function is not
        implemented.
        """

    @abstractmethod
    def y_0(self, x: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
        """
        Returns the initial value of y at the point in the spatial domain
        defined by x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """

    @abstractmethod
    def discrete_y_0(self) -> np.ndarray:
        """
        Returns the discretised initial values of y over a spatial mesh.
        """


class WellDefinedInitialCondition(InitialCondition):
    """
    An initial condition defined explicitly by a function.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            y_0_func: Callable[[Optional[Tuple[float, ...]]], np.ndarray]):
        """
        :param bvp: the boundary value problem to turn into an initial value
        problem by providing the initial conditions for it
        :param y_0_func: the initial value function that returns an array
        containing the values of y at the spatial coordinates defined by
        its input
        """
        self._bvp = bvp
        self._y_0_func = y_0_func

    def is_well_defined(self) -> bool:
        return True

    def y_0(self, x: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
        return self._y_0_func(x)

    def discrete_y_0(self) -> np.ndarray:
        diff_eq = self._bvp.differential_equation()
        if diff_eq.x_dimension():
            mesh = self._bvp.mesh()

            y_0 = np.empty(self._bvp.y_shape())
            for index in np.ndindex(mesh.shape()):
                y_0[(*index, slice(None))] = self._y_0_func(mesh.x(index))

            apply_constraints_along_last_axis(self._bvp.y_constraints(), y_0)
        else:
            y_0 = self._y_0_func(None)

        return y_0


class DiscreteInitialCondition(InitialCondition):
    """
    An initial condition defined by a fixed array of values.
    """

    def __init__(self, y_0: np.ndarray):
        """
        :param y_0: the array containing the initial values of y over a spatial
        mesh (which may be 0 dimensional in case of an ODE)
        """
        self._y_0 = np.copy(y_0)

    def is_well_defined(self) -> bool:
        return False

    def y_0(self, x: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
        pass

    def discrete_y_0(self) -> np.ndarray:
        return np.copy(self._y_0)


class GaussianInitialCondition(WellDefinedInitialCondition):
    """
    An initial condition defined explicitly by Gaussian functions.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            means_and_covs: Tuple[Tuple[np.ndarray, np.ndarray], ...],
            multipliers: Optional[np.ndarray] = None):
        """
        :param bvp: the boundary value problem to turn into an initial value
        problem by providing the initial conditions for it
        :param means_and_covs: a tuple of mean vectors and covariance matrices
        defining the multivariate Gaussian PDFs corresponding to each element
        of y_0
        :param multipliers: an array of multipliers for each element of the
        initial y values
        """
        diff_eq = bvp.differential_equation()
        assert diff_eq.x_dimension()
        assert len(means_and_covs) == diff_eq.y_dimension()
        for mean, cov in means_and_covs:
            assert mean.shape == (diff_eq.x_dimension(),)
            assert cov.shape == (diff_eq.x_dimension(), diff_eq.x_dimension())
        self._means_and_covs = deepcopy(means_and_covs)

        if multipliers is not None:
            assert multipliers.shape == (diff_eq.y_dimension(),)
            self._multipliers = np.copy(multipliers)
        else:
            self._multipliers = np.ones(diff_eq.y_dimension())

        super(GaussianInitialCondition, self).__init__(bvp, self._y_0_func)

    @staticmethod
    def multivariate_gaussian(
            x: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> float:
        """
        Returns the value of a Gaussian probability distribution function
        defined by the provided mean and covariance at the coordinates
        specified by x.

        :param x: the point at which the value of the PDF is to be calculated
        :param mean: the mean of the PDF
        :param cov: the covariance of the PDF
        :return: the value of the multivariate Gaussian PDF at x
        """
        centered_x = x - mean
        return 1. / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)) * \
            np.exp(-.5 * centered_x.T @ np.linalg.inv(cov) @ centered_x)

    def _y_0_func(self, x: Optional[Tuple[float, ...]]) -> np.ndarray:
        """
        Calculates and returns the values of the multivariate Gaussian PDFs
        corresponding to each element of y_0 at x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """
        x_arr = np.array(x)
        return np.array(
            [self.multivariate_gaussian(x_arr, mean, cov)
             for mean, cov in self._means_and_covs]) * self._multipliers
