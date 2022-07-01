from __future__ import annotations

from typing import Generator, List, NamedTuple, Optional, Sequence, Set

import numpy as np
from scipy.interpolate import interpn

from pararealml.constraint import apply_constraints_along_last_axis
from pararealml.differential_equation import NBodyGravitationalEquation
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.plot import (
    ContourPlot,
    NBodyPlot,
    PhaseSpacePlot,
    Plot,
    QuiverPlot,
    ScatterPlot,
    SpaceLinePlot,
    StreamPlot,
    SurfacePlot,
    TimePlot,
)


class Solution:
    """
    A solution to an IVP.
    """

    def __init__(
        self,
        ivp: InitialValueProblem,
        t_coordinates: np.ndarray,
        discrete_y: np.ndarray,
        vertex_oriented: Optional[bool] = None,
        d_t: Optional[float] = None,
    ):
        """
        :param ivp: the solved initial value problem
        :param t_coordinates: the time steps at which the solution is evaluated
        :param discrete_y: the solution to the IVP at the specified time steps
        :param vertex_oriented: whether the solution is vertex or cell oriented
            along the spatial domain; if the IVP is an ODE, it can be None
        :param d_t: the temporal step size of the solution; if it is None, it
            is inferred from the `t_coordinates` (which may lead to floating
            point issues)
        """
        if t_coordinates.ndim != 1:
            raise ValueError(
                f"number of t coordinate dimensions ({t_coordinates.ndim}) "
                "must be 1"
            )
        if len(t_coordinates) == 0:
            raise ValueError("length of t coordinates must be greater than 0")
        if (
            ivp.constrained_problem.differential_equation.x_dimension
            and vertex_oriented is None
        ):
            raise ValueError(
                "vertex orientation must be defined for solutions to PDEs"
            )
        y_shape = ivp.constrained_problem.y_shape(vertex_oriented)
        if discrete_y.shape != ((len(t_coordinates),) + y_shape):
            raise ValueError(
                "expected solution shape to be "
                f"{((len(t_coordinates),) + y_shape)} but got "
                f"{discrete_y.shape}"
            )

        self._ivp = ivp
        self._t_coordinates = np.copy(t_coordinates)
        self._discrete_y = np.copy(discrete_y)
        self._vertex_oriented = vertex_oriented

        self._t_coordinates.setflags(write=False)

        if d_t is None:
            d_t = (
                0.0
                if len(t_coordinates) == 1
                else t_coordinates[1] - t_coordinates[0]
            )
        self._d_t = d_t

    @property
    def initial_value_problem(self) -> InitialValueProblem:
        """
        The solved initial value problem.
        """
        return self._ivp

    @property
    def vertex_oriented(self) -> Optional[bool]:
        """
        Whether the solution is vertex or cell oriented along the spatial
        domain. If the solution is that of an ODE, it is None.
        """
        return self._vertex_oriented

    @property
    def d_t(self) -> float:
        """
        The temporal step size of the solution.
        """
        return self._d_t

    @property
    def t_coordinates(self) -> np.ndarray:
        """
        The time coordinates at which the solution is evaluated.
        """
        return self._t_coordinates

    def y(
        self,
        x: Optional[np.ndarray] = None,
        interpolation_method: str = "linear",
    ) -> np.ndarray:
        """
        Interpolates and returns the values of y at the specified
        spatial coordinates at every time step.

        :param x: the spatial coordinates with a shape of (..., x_dimension)
        :param interpolation_method: the interpolation method to use
        :return: the interpolated value of y at the provided spatial
            coordinates at every time step
        """
        cp = self._ivp.constrained_problem
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            return np.copy(self._discrete_y)

        y = interpn(
            cp.mesh.axis_coordinates(self._vertex_oriented),
            np.moveaxis(self._discrete_y, 0, -2),
            x,
            method=interpolation_method,
            bounds_error=False,
            fill_value=None,
        )
        y = np.moveaxis(y, -2, 0)
        y = y.reshape(
            (len(self._t_coordinates),) + x.shape[:-1] + (diff_eq.y_dimension,)
        )
        return np.ascontiguousarray(y)

    def discrete_y(
        self,
        vertex_oriented: Optional[bool] = None,
        interpolation_method: str = "linear",
    ) -> np.ndarray:
        """
        Returns the discrete solution evaluated either at vertices or the cell
        centers of the spatial mesh.

        :param vertex_oriented: whether the solution returned should be
            evaluated at the vertices or the cell centers of the spatial mesh;
            only interpolation is supported, therefore, it is not possible to
            evaluate the solution at the vertices based on a cell-oriented
            solution
        :param interpolation_method: the interpolation method to use
        :return: the discrete solution
        """
        if vertex_oriented is None:
            vertex_oriented = self._vertex_oriented

        cp = self._ivp.constrained_problem
        if (
            not cp.differential_equation.x_dimension
            or self._vertex_oriented == vertex_oriented
        ):
            return np.copy(self._discrete_y)

        x = cp.mesh.all_index_coordinates(vertex_oriented)
        discrete_y = self.y(x, interpolation_method)
        if vertex_oriented:
            apply_constraints_along_last_axis(
                cp.static_y_vertex_constraints, discrete_y
            )
        return discrete_y

    def diff(self, solutions: Sequence[Solution], atol: float = 1e-8) -> Diffs:
        """
        Calculates and returns the difference between the provided solutions
        and this solution at every matching time point across all solutions.

        :param solutions: the solutions to compare to
        :param atol: the maximum absolute difference between two time points
            considered to be matching
        :return: a `Diffs` instance containing a 1D array representing the
            matching time points and a sequence of sequence of arrays
            representing the differences between this solution and each of the
            provided solutions at the matching time points
        """
        if len(solutions) == 0:
            raise ValueError("length of solutions must be greater than 0")

        matching_time_points = []
        all_diffs: List[List[np.ndarray]] = []

        all_time_points = [self._t_coordinates]
        all_time_steps = [self._d_t]
        other_discrete_ys = []
        for solution in solutions:
            all_diffs.append([])
            all_time_points.append(solution.t_coordinates)
            all_time_steps.append(solution.d_t)
            other_discrete_ys.append(
                solution.discrete_y(self._vertex_oriented)
            )

        fewest_time_points_ind = 0
        fewest_time_points = None
        for i, time_points in enumerate(all_time_points):
            n_time_points = len(time_points)
            if (
                fewest_time_points is None
                or n_time_points < fewest_time_points
            ):
                fewest_time_points = n_time_points
                fewest_time_points_ind = i

        for i, t in enumerate(all_time_points[fewest_time_points_ind]):
            all_match = True
            indices_of_time_points = []

            for j, time_points in enumerate(all_time_points):
                if fewest_time_points_ind == j:
                    indices_of_time_points.append(i)
                    continue

                index_of_time_point = int(
                    round((t - time_points[0]) / all_time_steps[j])
                )
                if (
                    0 <= index_of_time_point < len(time_points)
                ) and np.isclose(
                    t, time_points[index_of_time_point], atol=atol, rtol=0.0
                ):
                    indices_of_time_points.append(index_of_time_point)
                else:
                    all_match = False
                    break

            if all_match:
                matching_time_points.append(t)

                for j, discrete_y in enumerate(other_discrete_ys):
                    diff = (
                        discrete_y[indices_of_time_points[j + 1]]
                        - self._discrete_y[indices_of_time_points[0]]
                    )

                    all_diffs[j].append(diff)

        matching_time_point_array = np.array(matching_time_points)
        diff_arrays = [np.array(diff) for diff in all_diffs]
        return Diffs(matching_time_point_array, diff_arrays)

    def generate_plots(self, **kwargs) -> Generator[Plot, None, None]:
        """
        Returns a generator for generating all applicable plots for the
        solution.

        :param kwargs: arguments to pass onto the generated plot objects
        :return: a generator for generating all plots
        """
        cp = self._ivp.constrained_problem
        diff_eq = cp.differential_equation

        if diff_eq.x_dimension > 3:
            return

        if diff_eq.x_dimension == 0:
            if isinstance(diff_eq, NBodyGravitationalEquation):
                yield NBodyPlot(self._discrete_y, diff_eq, **kwargs)
            else:
                yield TimePlot(self._discrete_y, self._t_coordinates, **kwargs)
                if 2 <= diff_eq.y_dimension <= 3:
                    yield PhaseSpacePlot(self._discrete_y, **kwargs)

        else:
            vector_index_set: Set[int] = set()
            if diff_eq.x_dimension > 1:
                all_vector_field_indices = diff_eq.all_vector_field_indices
                if all_vector_field_indices is not None:
                    for indices in all_vector_field_indices:
                        vector_index_set.update(indices)
                        vector_field = self._discrete_y[..., indices]
                        yield QuiverPlot(
                            vector_field,
                            cp.mesh,
                            self._vertex_oriented,
                            **kwargs,
                        )
                        if diff_eq.x_dimension == 2:
                            yield StreamPlot(
                                vector_field,
                                cp.mesh,
                                self._vertex_oriented,
                                **kwargs,
                            )

            for i in range(diff_eq.y_dimension):
                if i in vector_index_set:
                    continue

                scalar_field = self._discrete_y[..., i : i + 1]

                if diff_eq.x_dimension == 1:
                    yield SpaceLinePlot(
                        scalar_field, cp.mesh, self._vertex_oriented, **kwargs
                    )

                elif diff_eq.x_dimension == 2:
                    yield ContourPlot(
                        scalar_field, cp.mesh, self._vertex_oriented, **kwargs
                    )
                    yield SurfacePlot(
                        scalar_field, cp.mesh, self._vertex_oriented, **kwargs
                    )

                else:
                    yield ScatterPlot(
                        scalar_field, cp.mesh, self._vertex_oriented, **kwargs
                    )


class Diffs(NamedTuple):
    """
    A representation of the difference between a solution and one or more other
    solutions at time points that match across all solutions.
    """

    matching_time_points: np.ndarray
    differences: Sequence[np.ndarray]
