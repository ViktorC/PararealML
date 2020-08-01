from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import interpn

from src.core.constraint import apply_constraints_along_last_axis
from src.core.initial_value_problem import InitialValueProblem


class Solution:
    """
    A solution to an IVP.
    """

    def __init__(
            self,
            ivp: InitialValueProblem,
            t_coordinates: np.ndarray,
            discrete_y: np.ndarray,
            vertex_oriented: Optional[bool] = None):
        """
        :param ivp: the initial value problem solved
        :param t_coordinates: the time steps at which the solution is evaluated
        :param discrete_y: the solution to the IVP at the specified time steps
        :param vertex_oriented: whether the solution is vertex or cell oriented
        along the spatial domain; if the IVP is an ODE, it can be None
        """
        assert len(t_coordinates.shape) == 1
        assert len(t_coordinates) > 0
        assert discrete_y.shape == (
                (len(t_coordinates),) +
                ivp.boundary_value_problem.y_shape(vertex_oriented)
        )

        self._ivp = ivp
        self._t_coordinates = t_coordinates
        self._discrete_y = discrete_y
        self._vertex_oriented = vertex_oriented

    @property
    def initial_value_problem(self) -> InitialValueProblem:
        """
        Returns the IVP this object represents a solution to.
        """
        return self._ivp

    @property
    def vertex_oriented(self) -> Optional[bool]:
        """
        Returns whether the solution is vertex or cell oriented along the
        spatial domain. If the solution is that of an ODE, it returns None.
        """
        return self._vertex_oriented

    @property
    def t_coordinates(self) -> np.ndarray:
        """
        Returns the time coordinates at which the solution is evaluated.
        """
        return np.copy(self._t_coordinates)

    def x_coordinates(
            self,
            vertex_oriented: bool
    ) -> Optional[Sequence[np.ndarray]]:
        """
        Returns the spatial coordinates at which the solution is evaluated. To
        get the spatial coordinates of the discrete solution of this instance,
        call this function with the value of the vertex_oriented property.

        :param vertex_oriented: whether the coordinates of the vertices or the
        cell centers of the spatial mesh are to be returned
        :return: a tuple of arrays each representing the coordinates along the
        corresponding axis
        """
        return self._ivp \
            .boundary_value_problem \
            .mesh \
            .coordinates(vertex_oriented)

    def y(
            self,
            x: Optional[np.ndarray] = None,
            interpolation_method: Optional[str] = None
    ) -> np.ndarray:
        """
        Interpolates and returns the values of y at the specified
        spatial coordinates at every time step.

        :param x: the spatial coordinates with a shape of (..., x_dimension)
        :param interpolation_method: the interpolation method to use; if it is
        None, linear interpolation is used
        :return: the interpolated value of y at the provided spatial
        coordinates at every time step
        """
        if interpolation_method is None:
            interpolation_method = 'linear'

        bvp = self._ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        if diff_eq.x_dimension:
            assert x is not None
            assert x.shape[-1] == diff_eq.x_dimension

            y = interpn(
                bvp.mesh.coordinates(self._vertex_oriented),
                np.moveaxis(self._discrete_y, 0, -2),
                x,
                method=interpolation_method,
                bounds_error=False,
                fill_value=None)

            y = np.moveaxis(y, -2, 0)
            y = y.reshape(
                (len(self._t_coordinates),) +
                x.shape[:-1] +
                (diff_eq.y_dimension,))
            y = np.ascontiguousarray(y)
        else:
            y = np.copy(self._discrete_y)

        return y

    def discrete_y(
            self,
            vertex_oriented: Optional[bool] = None,
            interpolation_method: Optional[str] = None
    ) -> np.ndarray:
        """
        Returns the discrete solution evaluated either at vertices or the cell
        centers of the spatial mesh.

        :param vertex_oriented: whether the solution returned should be
        evaluated at the vertices or the cell centers of the spatial mesh; only
        interpolation is supported, therefore, it is not possible to evaluate
        the solution at the vertices based on a cell-oriented solution
        :param interpolation_method: the interpolation method to use; if it is
        None, linear interpolation is used
        :return: the discrete solution
        """
        bvp = self._ivp.boundary_value_problem

        if (bvp.differential_equation.x_dimension == 0) \
                or (self._vertex_oriented == vertex_oriented):
            return np.copy(self._discrete_y)

        coordinate_system = bvp.mesh.coordinates(vertex_oriented)
        mesh_grid = np.meshgrid(*coordinate_system, indexing='ij')
        coordinates = np.stack(mesh_grid, axis=-1)
        discrete_y = self.y(coordinates, interpolation_method)

        if vertex_oriented:
            apply_constraints_along_last_axis(
                bvp.y_vertex_constraints, discrete_y)

        return discrete_y
