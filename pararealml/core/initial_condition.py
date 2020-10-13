from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import Tuple, Optional, Callable, Sequence

import numpy as np

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.constraint import apply_constraints_along_last_axis
from pararealml.core.solution import Solution


class InitialCondition(ABC):
    """
    A base class for initial conditions.
    """

    @abstractmethod
    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        """
        Returns the initial value of y at the point in the spatial domain
        defined by x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """

    @abstractmethod
    def discrete_y_0(
            self,
            vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns the discretised initial values of y evaluated at the vertices
        or cell centers of the spatial mesh.

        :param vertex_oriented: whether the initial conditions are to be
            evaluated at the vertices or cell centers of the spatial mesh
        :return: the discretised initial values
        """


class DiscreteInitialCondition(InitialCondition):
    """
    An initial condition defined by a fixed array of values.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            y_0: np.ndarray,
            vertex_oriented: Optional[bool] = None,
            t_0: Optional[float] = None,
            interpolation_method: Optional[str] = None):
        """
        :param cp: the constrained problem to turn into an initial value
            problem by providing the initial conditions for it
        :param y_0: the array containing the initial values of y over a spatial
            mesh (which may be 0 dimensional in case of an ODE)
        :param vertex_oriented: whether the initial conditions are evaluated at
            the vertices or cell centers of the spatial mesh; it the
            constrained problem is an ODE, it can be None
        :param t_0: the lower boundary of the time domain
        :param interpolation_method: the interpolation method to use to
            calculate values that do not exactly fall on points of the y_0
            grid; if the constrained problem is based on an ODE, it can be None
        """
        self._interpolation_method = interpolation_method

        y_0_copy = np.copy(y_0)
        if vertex_oriented:
            apply_constraints_along_last_axis(
                cp.static_y_vertex_constraints
                if cp.are_all_boundary_conditions_static else
                cp.create_y_vertex_constraints(
                    cp.create_boundary_constraints(True, t_0)[0]),
                y_0_copy)
        y_0_copy = y_0_copy.reshape((1,) + y_0_copy.shape)

        self._y_0_solution = Solution(
            cp, np.zeros(1), y_0_copy, vertex_oriented)

    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        return self._y_0_solution.y(
            np.asarray(x), self._interpolation_method)[0]

    def discrete_y_0(
            self,
            vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        return self._y_0_solution.discrete_y(
            vertex_oriented, self._interpolation_method)[0]


class ContinuousInitialCondition(InitialCondition):
    """
    An initial condition defined by a function.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            y_0_func:
            Callable[[Optional[Sequence[float]]], Sequence[float]],
            t_0: Optional[float] = None):
        """
        :param cp: the constrained problem to turn into an initial value
            problem by providing the initial conditions for it
        :param y_0_func: the initial value function that returns an array
            containing the values of y at the spatial coordinates defined by
            its input
        :param t_0: the lower boundary of the time domain
        """
        self._cp = cp
        self._y_0_func = y_0_func
        self._discrete_y_0_vertices = self._create_discrete_y_0(True, t_0)
        self._discrete_y_0_cells = self._create_discrete_y_0(False, None)

    def y_0(self, x: Optional[Sequence[float]]) -> Sequence[float]:
        return self._y_0_func(x)

    def discrete_y_0(
            self,
            vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        return np.copy(
            self._discrete_y_0_vertices if vertex_oriented
            else self._discrete_y_0_cells)

    def _create_discrete_y_0(
            self,
            vertex_oriented: bool,
            t_0: Optional[float]
    ) -> np.ndarray:
        """
        Creates the discretised initial values of y evaluated at the vertices
        or cell centers of the spatial mesh.

        :param vertex_oriented: whether the initial conditions are to be
            evaluated at the vertices or cell centers of the spatial mesh
        :param t_0: the lower boundary of the time domain
        :return: the discretised initial values
        """
        diff_eq = self._cp.differential_equation
        if diff_eq.x_dimension:
            mesh = self._cp.mesh

            y_0 = np.empty(self._cp.y_shape(vertex_oriented))
            for index in np.ndindex(y_0.shape[:-1]):
                y_0[(*index, slice(None))] = self._y_0_func(
                    mesh.x(index, vertex_oriented))

            if vertex_oriented:
                apply_constraints_along_last_axis(
                    self._cp.static_y_vertex_constraints
                    if self._cp.are_all_boundary_conditions_static else
                    self._cp.create_y_vertex_constraints(
                        self._cp.create_boundary_constraints(True, t_0)[0]),
                    y_0)
        else:
            y_0 = self._y_0_func(None)

        return y_0


class GaussianInitialCondition(ContinuousInitialCondition):
    """
    An initial condition defined explicitly by Gaussian functions.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            means_and_covs: Sequence[Tuple[np.ndarray, np.ndarray]],
            multipliers: Optional[Sequence[float]] = None,
            t_0: Optional[float] = None):
        """
        :param cp: the constrained problem to turn into an initial value
            problem by providing the initial conditions for it
        :param means_and_covs: a sequence of tuples of mean vectors and
            covariance matrices defining the multivariate Gaussian PDFs
            corresponding to each element of y_0
        :param multipliers: an array of multipliers for each element of the
            initial y values
        :param t_0: the lower boundary of the time domain
        """
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            raise ValueError
        if len(means_and_covs) != diff_eq.y_dimension:
            raise ValueError
        for mean, cov in means_and_covs:
            if mean.shape != (diff_eq.x_dimension,):
                raise ValueError
            if cov.shape != (diff_eq.x_dimension, diff_eq.x_dimension):
                raise ValueError
        self._means_and_covs = deepcopy(means_and_covs)

        if multipliers is not None:
            if len(multipliers) != diff_eq.y_dimension:
                raise ValueError
            self._multipliers = copy(multipliers)
        else:
            self._multipliers = [1.] * diff_eq.y_dimension

        super(GaussianInitialCondition, self).__init__(cp, self._y_0_func, t_0)

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
