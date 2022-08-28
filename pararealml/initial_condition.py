from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interpn
from scipy.stats import beta, multivariate_normal

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.constraint import apply_constraints_along_last_axis
from pararealml.mesh import to_cartesian_coordinates

VectorizedInitialConditionFunction = Callable[
    [Optional[np.ndarray]], np.ndarray
]


class InitialCondition(ABC):
    """
    A base class for initial conditions.
    """

    @abstractmethod
    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns the initial value of y at the points in the spatial domain
        defined by x.

        :param x: a 2D array (n, x_dimension) of the spatial coordinates for
            PDEs and None for ODEs
        :return: a 2D array (n, y_dimension) of the initial value of y at the
            coordinates for PDEs and a 1D array (y_dimension) for ODEs
        """

    @abstractmethod
    def discrete_y_0(
        self, vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns the discretized initial values of y evaluated at the vertices
        or cell centers of the spatial mesh.

        :param vertex_oriented: whether the initial conditions are to be
            evaluated at the vertices or cell centers of the spatial mesh
        :return: the discretized initial values
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
        interpolation_method: str = "linear",
    ):
        """
        :param cp: the constrained problem to provide initial conditions for
        :param y_0: the array containing the initial values of y over a spatial
            mesh (which may be 0 dimensional in case of an ODE)
        :param vertex_oriented: whether the initial conditions are evaluated at
            the vertices or cell centers of the spatial mesh; it the
            constrained problem is an ODE, it can be None
        :param interpolation_method: the interpolation method to use to
            calculate values that do not exactly fall on points of the y_0
            grid; if the constrained problem is based on an ODE, it can be None
        """
        if cp.differential_equation.x_dimension and vertex_oriented is None:
            raise ValueError("vertex orientation must be defined for PDEs")
        if y_0.shape != cp.y_shape(vertex_oriented):
            raise ValueError(
                f"discrete initial value shape {y_0.shape} must match "
                "constrained problem solution shape "
                f"{cp.y_shape(vertex_oriented)}"
            )

        self._cp = cp
        self._y_0 = np.copy(y_0)
        self._vertex_oriented = vertex_oriented
        self._interpolation_method = interpolation_method

        if vertex_oriented:
            apply_constraints_along_last_axis(
                cp.static_y_vertex_constraints, self._y_0
            )

    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        if not self._cp.differential_equation.x_dimension:
            return np.copy(self._y_0)

        return interpn(
            self._cp.mesh.axis_coordinates(self._vertex_oriented),
            self._y_0,
            x,
            method=self._interpolation_method,
            bounds_error=False,
            fill_value=None,
        )

    def discrete_y_0(
        self, vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        if vertex_oriented is None:
            vertex_oriented = self._vertex_oriented

        if (
            not self._cp.differential_equation.x_dimension
            or vertex_oriented == self._vertex_oriented
        ):
            return np.copy(self._y_0)

        y_0 = self.y_0(self._cp.mesh.all_index_coordinates(vertex_oriented))
        if vertex_oriented:
            apply_constraints_along_last_axis(
                self._cp.static_y_vertex_constraints, y_0
            )
        return y_0


class ConstantInitialCondition(DiscreteInitialCondition):
    """
    An initial condition defined by a sequence of scalars each denoting the
    constant initial value of the corresponding element of y.
    """

    def __init__(self, cp: ConstrainedProblem, constant_y_0s: Sequence[float]):
        """
        :param cp: the constrained problem to provide initial conditions for
        :param constant_y_0s: the constant initial values of the components of
            y (at every point of the mesh if the constrained problem is a PDE)
        """
        y_dim = cp.differential_equation.y_dimension
        if len(constant_y_0s) != y_dim:
            raise ValueError(
                f"length of constant y0 values ({len(constant_y_0s)}) must "
                f"match number of y components ({y_dim})"
            )

        ic = np.empty(cp.y_shape(True))
        for i, y_0 in enumerate(constant_y_0s):
            ic[..., i] = y_0

        super(ConstantInitialCondition, self).__init__(cp, ic, True)


class ContinuousInitialCondition(InitialCondition):
    """
    An initial condition defined by a function.
    """

    def __init__(
        self,
        cp: ConstrainedProblem,
        y_0_func: VectorizedInitialConditionFunction,
        multipliers: Optional[Sequence[float]] = None,
    ):
        """
        :param cp: the constrained problem to provide initial conditions for
        :param y_0_func: the initial value function that returns an array
            containing the values of y at the spatial coordinates defined by
            its input
        :param multipliers: an array of multipliers for each element of the
            initial y values
        """
        diff_eq = cp.differential_equation
        if multipliers is not None:
            if len(multipliers) != diff_eq.y_dimension:
                raise ValueError(
                    f"length of multipliers ({len(multipliers)}) must match "
                    f"number of y dimensions ({diff_eq.y_dimension})"
                )
            self._multipliers = np.array(multipliers)
        else:
            self._multipliers = np.ones(diff_eq.y_dimension)

        self._cp = cp
        self._y_0_func = y_0_func
        self._discrete_y_0_vertices = self._create_discrete_y_0(True)
        self._discrete_y_0_cells = self._create_discrete_y_0(False)

    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        return np.multiply(self._y_0_func(x), self._multipliers)

    def discrete_y_0(
        self, vertex_oriented: Optional[bool] = None
    ) -> np.ndarray:
        return np.copy(
            self._discrete_y_0_vertices
            if vertex_oriented
            else self._discrete_y_0_cells
        )

    def _create_discrete_y_0(self, vertex_oriented: bool) -> np.ndarray:
        """
        Creates the discretized initial values of y evaluated at the vertices
        or cell centers of the spatial mesh.

        :param vertex_oriented: whether the initial conditions are to be
            evaluated at the vertices or cell centers of the spatial mesh
        :return: the discretized initial values
        """
        diff_eq = self._cp.differential_equation
        if not diff_eq.x_dimension:
            y_0 = np.array(self.y_0(None))
            if y_0.shape != self._cp.y_shape():
                raise ValueError(
                    "expected initial condition function output shape to be "
                    f"{self._cp.y_shape()} but got {y_0.shape}"
                )

            return y_0

        x = self._cp.mesh.all_index_coordinates(vertex_oriented, flatten=True)
        y_0 = self.y_0(x)
        if y_0.shape != (len(x), diff_eq.y_dimension):
            raise ValueError(
                "expected initial condition function output shape to be "
                f"{(len(x), diff_eq.y_dimension)} but got {y_0.shape}"
            )

        y_0 = y_0.reshape(self._cp.y_shape(vertex_oriented))
        if vertex_oriented:
            apply_constraints_along_last_axis(
                self._cp.static_y_vertex_constraints, y_0
            )
        return y_0

    def _convert_coordinates_to_cartesian(self, x: np.ndarray) -> np.ndarray:
        """
        Converts the provided coordinates to Cartesian coordinates.

        :param x: the coordinates to convert
        :return: the converted Cartesian coordinates
        """
        cartesian_x = to_cartesian_coordinates(
            [x[:, i] for i in range(x.shape[1])],
            self._cp.mesh.coordinate_system_type,
        )
        return np.stack(cartesian_x, axis=-1)


class GaussianInitialCondition(ContinuousInitialCondition):
    """
    An initial condition defined by a Gaussian probability density function.
    """

    def __init__(
        self,
        cp: ConstrainedProblem,
        means_and_covs: Sequence[Tuple[np.ndarray, np.ndarray]],
        multipliers: Optional[Sequence[float]] = None,
    ):
        """
        :param cp: the constrained problem to provide initial conditions for
        :param means_and_covs: a sequence of tuples of mean vectors and
            covariance matrices defining the multivariate Gaussian PDFs
            corresponding to each element of y_0
        :param multipliers: an array of multipliers for each element of the
            initial y values
        """
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            raise ValueError("constrained problem must be a PDE")
        if len(means_and_covs) != diff_eq.y_dimension:
            raise ValueError(
                f"number of means and covariances ({len(means_and_covs)}) "
                f"must match number of y dimensions ({diff_eq.y_dimension})"
            )
        for mean, cov in means_and_covs:
            if mean.shape != (diff_eq.x_dimension,):
                raise ValueError(
                    f"expected mean shape to be {(diff_eq.x_dimension,)} but "
                    f"got {mean.shape}"
                )
            if cov.shape != (diff_eq.x_dimension, diff_eq.x_dimension):
                raise ValueError(
                    "expected covariance shape to be "
                    f"{(diff_eq.x_dimension, diff_eq.x_dimension)} but got "
                    f"{cov.shape}"
                )

        self._means_and_covs = deepcopy(means_and_covs)

        super(GaussianInitialCondition, self).__init__(
            cp, self._y_0, multipliers
        )

    def _y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        """
        Calculates and returns the values of the multivariate Gaussian PDFs
        corresponding to each element of y_0 at x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """
        cartesian_x = self._convert_coordinates_to_cartesian(x)
        y_0 = np.empty((len(x), self._cp.differential_equation.y_dimension))
        for i in range(self._cp.differential_equation.y_dimension):
            mean, cov = self._means_and_covs[i]
            y_0[:, i] = multivariate_normal.pdf(
                cartesian_x, mean=mean, cov=cov
            )

        return y_0


class MarginalBetaProductInitialCondition(ContinuousInitialCondition):
    """
    An initial condition defined by the product of marginal Beta probability
    density functions.
    """

    def __init__(
        self,
        cp: ConstrainedProblem,
        all_alphas_and_betas: Sequence[Sequence[Tuple[float, float]]],
        multipliers: Optional[Sequence[float]] = None,
    ):
        """
        :param cp: the constrained problem to provide initial conditions for
        :param all_alphas_and_betas: a sequence (with an entry for each element
            of y) of sequences (with an entry for each spatial dimension) of
            tuples containing alpha and beta, the two parameters defining the
            beta distribution of the initial values of the corresponding
            element of y along the corresponding spatial axis
        :param multipliers: an array of multipliers for each element of y
        """
        diff_eq = cp.differential_equation
        if len(all_alphas_and_betas) != diff_eq.y_dimension:
            raise ValueError(
                "number of alphas and betas sequences "
                f"({len(all_alphas_and_betas)}) must match the number of y "
                f"dimensions ({diff_eq.y_dimension})"
            )

        if any(
            [
                len(alphas_and_betas) != diff_eq.x_dimension
                for alphas_and_betas in all_alphas_and_betas
            ]
        ):
            raise ValueError(
                "all sequences of alphas and betas must have same length as "
                f"number of spatial dimensions ({diff_eq.x_dimension})"
            )

        self._all_alphas_and_betas = deepcopy(all_alphas_and_betas)

        super(MarginalBetaProductInitialCondition, self).__init__(
            cp, self._y_0, multipliers
        )

    def _y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        """
        Calculates and returns the values of the products of the beta PDFs
        corresponding to each axis of x for each element of y_0 at x.

        :param x: the spatial coordinates
        :return: the initial value of y at the coordinates
        """
        cartesian_x = self._convert_coordinates_to_cartesian(x)
        return np.concatenate(
            [
                np.prod(
                    [
                        beta.pdf(cartesian_x[:, x_ind : x_ind + 1], a, b)
                        for x_ind, (a, b) in enumerate(alpha_and_betas)
                    ],
                    axis=0,
                )
                for alpha_and_betas in self._all_alphas_and_betas
            ],
            axis=-1,
        )


def vectorize_ic_function(
    ic_function: Callable[[Optional[Sequence[float]]], Sequence[float]]
) -> VectorizedInitialConditionFunction:
    """
    Vectorizes an initial condition function that operates on a single
    coordinate sequence so that it can operate on an array of coordinate
    sequences.

    The implementation of the vectorized function is nothing more than a for
    loop over the rows of coordinate sequences in the x argument.

    :param ic_function: the non-vectorized initial condition function
    :return: the vectorized initial condition function
    """

    def vectorized_ic_function(x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return np.array(ic_function(None))

        values = []
        for i in range(len(x)):
            values.append(ic_function(x[i]))
        return np.array(values)

    return vectorized_ic_function
