from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import Tuple, Optional, Callable, Sequence

import numpy as np
from scipy.interpolate import interpn

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.constraint import apply_constraints_along_last_axis


class InitialCondition(ABC):
    """
    A base class for initial conditions.
    """

    @property
    @abstractmethod
    def discrete_y_0(self) -> np.ndarray:
        """
        Returns the discretised initial values of y over a spatial mesh.
        """

    @abstractmethod
    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        """
        Returns the initial value of y at the point in the spatial domain
        defined by x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """


class DiscreteInitialCondition(InitialCondition):
    """
    An initial condition defined by a fixed array of values.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            y_0: np.ndarray,
            interpolation_method: str = 'linear'):
        """
        :param bvp: the boundary value problem to turn into an initial value
        problem by providing the initial conditions for it
        :param y_0: the array containing the initial values of y over a spatial
        mesh (which may be 0 dimensional in case of an ODE)
        :param interpolation_method: the interpolation method to use to
        calculate values that do not exactly fall on points of the y_0 grid
        """
        assert y_0.shape == bvp.y_shape

        self._bvp = bvp
        self._y_0 = np.copy(y_0)
        apply_constraints_along_last_axis(bvp.y_constraints, self._y_0)

        self._interpolation_method = interpolation_method
        self._x_coordinates = self._create_x_coordinates()

    @property
    def discrete_y_0(self) -> np.ndarray:
        return np.copy(self._y_0)

    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        return interpn(
            self._x_coordinates,
            self._y_0,
            np.asarray(x),
            method=self._interpolation_method)[0, ...]

    def _create_x_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Creates a tuple of arrays representing the coordinates along each axis
        of the mesh.
        """
        mesh = self._bvp.mesh
        mesh_shape = mesh.shape
        x_intervals = mesh.x_intervals

        x = []
        for axis in range(self._bvp.differential_equation.x_dimension):
            x_interval = x_intervals[axis]
            x.append(
                np.linspace(x_interval[0], x_interval[1], mesh_shape[axis]))

        return tuple(x)


class ContinuousInitialCondition(InitialCondition):
    """
    An initial condition defined by a function.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            y_0_func:
            Callable[[Optional[Sequence[float]]], Sequence[float]]):
        """
        :param bvp: the boundary value problem to turn into an initial value
        problem by providing the initial conditions for it
        :param y_0_func: the initial value function that returns an array
        containing the values of y at the spatial coordinates defined by
        its input
        """
        self._bvp = bvp
        self._y_0_func = y_0_func
        self._discrete_y_0 = self._create_discrete_y_0()

    @property
    def discrete_y_0(self) -> np.ndarray:
        return np.copy(self._discrete_y_0)

    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        return self._y_0_func(x)

    def _create_discrete_y_0(self) -> np.ndarray:
        """
        Creates the discretised initial values of y over the spatial mesh of
        the BVP associated with the initial condition.
        """
        diff_eq = self._bvp.differential_equation
        if diff_eq.x_dimension:
            mesh = self._bvp.mesh

            y_0 = np.empty(self._bvp.y_shape)
            for index in np.ndindex(mesh.shape):
                y_0[(*index, slice(None))] = self._y_0_func(mesh.x(index))

            apply_constraints_along_last_axis(self._bvp.y_constraints, y_0)
        else:
            y_0 = self._y_0_func(None)

        return y_0


class GaussianInitialCondition(ContinuousInitialCondition):
    """
    An initial condition defined explicitly by Gaussian functions.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            means_and_covs: Tuple[Tuple[np.ndarray, np.ndarray], ...],
            multipliers: Optional[Sequence[float]] = None):
        """
        :param bvp: the boundary value problem to turn into an initial value
        problem by providing the initial conditions for it
        :param means_and_covs: a tuple of mean vectors and covariance matrices
        defining the multivariate Gaussian PDFs corresponding to each element
        of y_0
        :param multipliers: an array of multipliers for each element of the
        initial y values
        """
        diff_eq = bvp.differential_equation
        assert diff_eq.x_dimension
        assert len(means_and_covs) == diff_eq.y_dimension
        for mean, cov in means_and_covs:
            assert mean.shape == (diff_eq.x_dimension,)
            assert cov.shape == (diff_eq.x_dimension, diff_eq.x_dimension)
        self._means_and_covs = deepcopy(means_and_covs)

        if multipliers is not None:
            assert len(multipliers) == diff_eq.y_dimension
            self._multipliers = copy(multipliers)
        else:
            self._multipliers = [1.] * diff_eq.y_dimension

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

    def _y_0_func(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        """
        Calculates and returns the values of the multivariate Gaussian PDFs
        corresponding to each element of y_0 at x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """
        x_arr = np.array(x)
        return [self.multivariate_gaussian(x_arr, mean, cov) *
                self._multipliers[i] for i, (mean, cov) in
                enumerate(self._means_and_covs)]
