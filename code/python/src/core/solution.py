from __future__ import annotations

from typing import Optional, Sequence, Any, NamedTuple

import numpy as np
from scipy.interpolate import interpn

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.constraint import apply_constraints_along_last_axis


class Solution:
    """
    A solution to an IVP.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            t_coordinates: np.ndarray,
            discrete_y: np.ndarray,
            vertex_oriented: Optional[bool] = None):
        """
        :param bvp: the boundary value problem that the initial value problem
        solved is based on
        :param t_coordinates: the time steps at which the solution is evaluated
        :param discrete_y: the solution to the IVP at the specified time steps
        :param vertex_oriented: whether the solution is vertex or cell oriented
            along the spatial domain; if the IVP is an ODE, it can be None
        """
        assert len(t_coordinates.shape) == 1
        assert len(t_coordinates) > 0
        assert discrete_y.shape == \
            ((len(t_coordinates),) + bvp.y_shape(vertex_oriented))

        self._bvp = bvp
        self._t_coordinates = t_coordinates
        self._discrete_y = discrete_y
        self._vertex_oriented = vertex_oriented

    @property
    def boundary_value_problem(self):
        """
        Returns the BVP that the IVP whose solution this object represents is
        based on.
        """
        return self._bvp

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
        return self._bvp.mesh.coordinates(vertex_oriented)

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

        diff_eq = self._bvp.differential_equation

        if diff_eq.x_dimension:
            assert x is not None
            assert x.shape[-1] == diff_eq.x_dimension

            y = interpn(
                self._bvp.mesh.coordinates(self._vertex_oriented),
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
            evaluated at the vertices or the cell centers of the spatial mesh;
            only interpolation is supported, therefore, it is not possible to
            evaluate the solution at the vertices based on a cell-oriented
            solution
        :param interpolation_method: the interpolation method to use; if it is
            None, linear interpolation is used
        :return: the discrete solution
        """
        if (self._bvp.differential_equation.x_dimension == 0) \
                or (self._vertex_oriented == vertex_oriented):
            return np.copy(self._discrete_y)

        coordinate_system = self._bvp.mesh.coordinates(vertex_oriented)
        mesh_grid = np.meshgrid(*coordinate_system, indexing='ij')
        coordinates = np.stack(mesh_grid, axis=-1)
        discrete_y = self.y(coordinates, interpolation_method)

        if vertex_oriented:
            apply_constraints_along_last_axis(
                self._bvp.y_vertex_constraints, discrete_y)

        return discrete_y

    def diff(
            self,
            solutions: Sequence[Solution],
            atol: float = 1e-8
    ) -> Diffs:
        """
        Calculates and returns the difference between the provided solutions
        and this solution at every matching time point across all solutions.

        :param solutions: the solutions to compare to
        :param atol: the maximum absolute difference between two time points
            considered to be matching
        :return: a `Diffs` instance containing a 1D array
            representing the matching time points and a sequence of sequence of
            arrays representing the differences between this solution and each
            of the provided solutions at the matching time points
        """
        assert len(solutions) > 0

        matching_time_points = []
        all_diffs = []

        all_time_points = [self._t_coordinates]
        other_discrete_ys = []
        for solution in solutions:
            all_diffs.append([])
            all_time_points.append(solution.t_coordinates)
            other_discrete_ys.append(
                solution.discrete_y(self._vertex_oriented))

        fewest_time_points_ind = 0
        fewest_time_points = None
        for i, time_points in enumerate(all_time_points):
            n_time_points = len(time_points)
            if fewest_time_points is None \
                    or n_time_points < fewest_time_points:
                fewest_time_points = n_time_points
                fewest_time_points_ind = i

        for i, t in enumerate(all_time_points[fewest_time_points_ind]):
            all_match = True
            indices_of_time_points = []

            for j, time_points in enumerate(all_time_points):
                if time_points is None:
                    continue

                if i == j:
                    indices_of_time_points.append(i)
                    continue

                insertion_ind = np.searchsorted(time_points, t)
                next_ind = min(len(time_points) - 1, insertion_ind + 1)

                if insertion_ind == next_ind:
                    all_time_points[j] = None
                else:
                    all_time_points[j] = all_time_points[j][next_ind:]

                if np.isclose(
                        t, time_points[insertion_ind], atol=atol, rtol=0.):
                    indices_of_time_points.append(insertion_ind)
                elif np.isclose(
                        t, time_points[next_ind], atol=atol, rtol=0.):
                    indices_of_time_points.append(next_ind)
                else:
                    all_match = False
                    break

            if all_match:
                matching_time_points.append(
                    self._t_coordinates[indices_of_time_points[0]])
                for j, discrete_y in enumerate(other_discrete_ys):
                    all_diffs[j].append(
                        discrete_y[indices_of_time_points[j + 1]] -
                        self._discrete_y[indices_of_time_points[0]])

        matching_time_point_array = np.array(matching_time_points)
        diff_arrays = [np.array(diff) for diff in all_diffs]
        return Diffs(matching_time_point_array, diff_arrays)

    def plot(self, solution_name: str, **kwargs: Any):
        """
        Plots the solution and saves it to a file.

        :param solution_name: the name of the solution; this is included in the
            file name of the saved plot
        :param kwargs: plotting configuration;
            see :func:`~src.utils.plot.plot_ivp_solution`
        """
        from src.utils.plot import plot_ivp_solution

        plot_ivp_solution(self, solution_name, **kwargs)


class Diffs(NamedTuple):
    """
    A representation of the difference between a solution and one or more other
    solutions at time points that match across all solutions.
    """
    matching_time_points: np.ndarray
    differences: Sequence[np.ndarray]
