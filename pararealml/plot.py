from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

from pararealml.differential_equation import NBodyGravitationalEquation
from pararealml.mesh import CoordinateSystem, Mesh


class Plot:
    """
    A base class for plots of the solutions of differential equations.
    """

    def __init__(self, figure: Figure):
        """
        :param figure: the figure of the plotted solution
        """
        self._figure = figure

    def show(self) -> Plot:
        """
        Displays the plot.

        If there are any other instantiated and unclosed plot objects, invoking
        this method displays those plots as well.

        Invoking the :func:`~pararealml.plot.Plot.save` method after invoking
        this one results in undefined behaviour since the plot may get closed
        as a side effect of this method.

        :return: the plot object the method is invoked on
        """
        plt.show()
        return self

    def save(self, file_path: str, extension: str = "png", **kwargs) -> Plot:
        """
        Saves the plot to the file system.

        Invoking this method after invoking :func:`~pararealml.plot.Plot.show`
        results in undefined behaviour.

        :param file_path: the path to save the image file to excluding any
            extensions
        :param extension: the file extension to use
        :param kwargs: any extra arguments
        :return: the plot object the method is invoked on
        """
        self._figure.savefig(f"{file_path}.{extension}", **kwargs)
        return self

    def close(self):
        """
        Closes the plot.
        """
        plt.close(self._figure)


class AnimatedPlot(Plot):
    """
    A base class for animated plots of the solutions of differential equations.
    """

    def __init__(
        self,
        figure: Figure,
        init_func: Callable[[], None],
        update_func: Callable[[int], None],
        n_time_steps: int,
        n_frames: int,
        interval: int,
    ):
        """
        :param figure: the figure of the plotted solution
        :param init_func: the animation initialization function
        :param update_func: the animation update function
        :param n_time_steps: the total number of time steps included in the
            solution
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        """
        super(AnimatedPlot, self).__init__(figure)
        time_steps = np.linspace(0, n_time_steps - 1, n_frames, dtype=int)
        self._animation = FuncAnimation(
            figure,
            func=update_func,
            init_func=init_func,
            frames=time_steps,
            interval=interval,
        )

    def save(self, file_path: str, extension: str = "gif", **kwargs) -> Plot:
        self._animation.save(f"{file_path}.{extension}", **kwargs)
        return self

    @staticmethod
    def _verify_pde_solution_shape_matches_problem(
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        expected_x_dims: Union[int, Tuple[int, int]],
        is_vector_field: bool,
    ):
        """
        Verifies that the shape of the input array representing the solution
        of a partial differential equation over the provided mesh and with the
        specified vertex orientation matches expectations.

        :param y: an array representing the solution of the partial
            differential equation
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param expected_x_dims: the expected number of spatial dimensions
        :param is_vector_field: whether the solution is supposed to be a vector
            field or a scalar field
        """
        if isinstance(expected_x_dims, int):
            if mesh.dimensions != expected_x_dims:
                raise ValueError(f"mesh must be {expected_x_dims} dimensional")
        elif not (expected_x_dims[0] <= mesh.dimensions <= expected_x_dims[1]):
            raise ValueError(
                f"mesh must be between {expected_x_dims[0]} and "
                f"{expected_x_dims[1]} dimensional"
            )

        if y.ndim != mesh.dimensions + 2:
            raise ValueError(
                f"number of y axes ({y.ndim}) must be two larger than mesh "
                f"dimensions ({mesh.dimensions})"
            )

        if y.shape[1:-1] != mesh.shape(vertex_oriented):
            raise ValueError(
                f"y shape {y.shape} must be compatible with mesh shape "
                f"{mesh.shape(vertex_oriented)}"
            )

        if is_vector_field:
            if y.shape[-1] != mesh.dimensions:
                raise ValueError(
                    f"number of y components ({y.shape[-1]}) must match "
                    f"x dimensions {mesh.dimensions}"
                )
        elif y.shape[-1] != 1:
            raise ValueError(
                f"number of y components ({y.shape[-1]}) must be one"
            )


class TimePlot(Plot):
    """
    A simple y against t plot to visualize the solutions of systems of ordinary
    differential equations.
    """

    def __init__(
        self,
        y: np.ndarray,
        t: np.ndarray,
        legend_location: Optional[str] = None,
        **_,
    ):
        """
        :param y: an array representing the solution of the ordinary
            differential equation system
        :param t: the time coordinates of the solution
        :param legend_location: the location of the legend denoting which graph
            represents which component of the solution
        :param _: any ignored extra arguments
        """
        if y.ndim != 2:
            raise ValueError(f"number of y axes ({y.ndim}) must be 2")
        if t.ndim != 1:
            raise ValueError(f"number of t axes ({t.ndim}) must be 1")
        if y.shape[0] != t.shape[0]:
            raise ValueError(
                f"first axis of y ({y.shape[0]}) must match length of t "
                f"({t.shape[0]})"
            )

        fig, ax = plt.subplots()

        for i in range(y.shape[1]):
            ax.plot(t, y[:, i], label=f"y{i}")

        ax.set_xlabel("t")
        ax.set_ylabel("y")
        if legend_location is not None:
            ax.legend(loc=legend_location)

        fig.tight_layout()
        super(TimePlot, self).__init__(fig)


class PhaseSpacePlot(Plot):
    """
    A phase-space plot to visualize the solutions of systems of two or three
    ordinary differential equations.
    """

    def __init__(self, y: np.ndarray, **_):
        """
        :param y: an array representing the solution of the ordinary
            differential equation system
        :param _: any ignored extra arguments
        """
        if y.ndim != 2:
            raise ValueError(f"number of y axes ({y.ndim}) must be 2")
        if not 2 <= y.shape[1] <= 3:
            raise ValueError(
                f"number of y components ({y.shape[1]}) must be either 2 or 3"
            )

        fig = plt.figure()

        if y.shape[1] == 2:
            ax = fig.add_subplot()
            ax.plot(y[:, 0], y[:, 1])
            ax.set_xlabel("y0")
            ax.set_ylabel("y1")
            ax.axis("equal")

        else:
            ax = fig.add_subplot(projection="3d")
            ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
            ax.set_xlabel("y0")
            ax.set_ylabel("y1")
            ax.set_zlabel("y2")
            ax.set_box_aspect(
                (np.ptp(y[:, 0]), np.ptp(y[:, 1]), np.ptp(y[:, 2]))
            )

        super(PhaseSpacePlot, self).__init__(fig)


class NBodyPlot(AnimatedPlot):
    """
    A 2D or 3D animated scatter plot to visualize the solutions of 2D or 3D
    n-body gravitational simulations.
    """

    def __init__(
        self,
        y: np.ndarray,
        diff_eq: NBodyGravitationalEquation,
        n_frames: int = 100,
        interval: int = 100,
        color_map: Colormap = cm.cividis,
        smallest_marker_size: float = 10.0,
        draw_trajectory: bool = True,
        trajectory_line_style: str = ":",
        trajectory_line_width: float = 0.5,
        span_scaling_factor: float = 0.25,
        **_,
    ):
        """
        :param y: an array representing the solution of the n-body
            gravitational differential equation
        :param diff_eq: the n-body gravitational differential equation solved
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param color_map: the color map to use for coloring the celestial
            objects
        :param smallest_marker_size: the size of the marker representing the
            smallest mass
        :param draw_trajectory: whether the trajectory of the objects should be
            plotted as well
        :param trajectory_line_style: the style of the trajectory line
        :param trajectory_line_width: the width of the trajectory line
        :param span_scaling_factor: the fraction of the peak-to-peak value of
            the object positions along each axis to pad the axis limits with;
            for example if the lowest and highest x coordinates of any object
            are -5 and 5 respectively, the limits of the x-axis are set at
            -5 - 10 * span_scaling_factor and 5 + 10 * span_scaling_factor
            respectively
        :param _: any ignored extra arguments
        """
        if y.ndim != 2:
            raise ValueError(f"number of y axes ({y.ndim}) must be 2")
        if y.shape[1] != diff_eq.y_dimension:
            raise ValueError(
                f"number of y components ({y.ndim}) must match differential "
                f"equation y dimension ({diff_eq.y_dimension})"
            )

        n_obj = diff_eq.n_objects
        n_obj_by_dims = n_obj * diff_eq.spatial_dimension

        x_coordinates = y[:, : n_obj_by_dims : diff_eq.spatial_dimension]
        y_coordinates = y[:, 1 : n_obj_by_dims : diff_eq.spatial_dimension]

        x_max = x_coordinates.max()
        x_min = x_coordinates.min()
        y_max = y_coordinates.max()
        y_min = y_coordinates.min()

        x_span = x_max - x_min
        y_span = y_max - y_min

        x_max += span_scaling_factor * x_span
        x_min -= span_scaling_factor * x_span
        y_max += span_scaling_factor * y_span
        y_min -= span_scaling_factor * y_span

        masses = np.asarray(diff_eq.masses)
        scaled_masses = (smallest_marker_size / np.min(masses)) * masses
        radii = np.power(3.0 * scaled_masses / (4.0 * np.pi), 1.0 / 3.0)
        marker_sizes = np.power(radii, 2) * np.pi

        colors = color_map(np.linspace(0.0, 1.0, n_obj))

        self._scatter_plot: Optional[PathCollection] = None
        self._line_plots: Optional[List[Union[Line2D, Line3D]]] = None

        style = "dark_background"

        with plt.style.context(style):
            fig = plt.figure()
            ax = fig.add_subplot(
                projection="3d" if diff_eq.spatial_dimension == 3 else None
            )

        if diff_eq.spatial_dimension == 2:
            coordinates = np.stack((x_coordinates, y_coordinates), axis=2)

            def init_plot():
                with plt.style.context(style):
                    ax.clear()
                    self._scatter_plot = ax.scatter(
                        x_coordinates[0, :],
                        y_coordinates[0, :],
                        s=marker_sizes,
                        c=colors,
                    )

                    if draw_trajectory:
                        self._line_plots = []
                        for i in range(n_obj):
                            self._line_plots.append(
                                ax.plot(
                                    x_coordinates[:1, i],
                                    y_coordinates[:1, i],
                                    color=colors[i],
                                    linestyle=trajectory_line_style,
                                    linewidth=trajectory_line_width,
                                )[0]
                            )

                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.axis("scaled")
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

            def update_plot(time_step: int):
                self._scatter_plot.set_offsets(coordinates[time_step, ...])
                if draw_trajectory:
                    for i in range(n_obj):
                        line_plot = self._line_plots[i]
                        line_plot.set_xdata(x_coordinates[: time_step + 1, i])
                        line_plot.set_ydata(y_coordinates[: time_step + 1, i])

        else:
            z_coordinates = y[:, 2:n_obj_by_dims:3]
            z_max = z_coordinates.max()
            z_min = z_coordinates.min()
            z_span = z_max - z_min
            z_max += span_scaling_factor * z_span
            z_min -= span_scaling_factor * z_span

            def init_plot():
                with plt.style.context(style):
                    ax.clear()
                    self._scatter_plot = ax.scatter(
                        x_coordinates[0, :],
                        y_coordinates[0, :],
                        z_coordinates[0, :],
                        s=marker_sizes,
                        c=colors,
                        depthshade=False,
                    )

                    if draw_trajectory:
                        self._line_plots = []
                        for i in range(n_obj):
                            self._line_plots.append(
                                ax.plot(
                                    x_coordinates[:1, i],
                                    y_coordinates[:1, i],
                                    z_coordinates[:1, i],
                                    color=colors[i],
                                    linestyle=trajectory_line_style,
                                    linewidth=trajectory_line_width,
                                )[0]
                            )

                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_zlim(z_min, z_max)
                    ax.set_box_aspect(
                        (x_max - x_min, y_max - y_min, z_max - z_min)
                    )
                    ax.set_facecolor("black")
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.grid(False)

            def update_plot(time_step: int):
                self._scatter_plot._offsets3d = (
                    x_coordinates[time_step, ...],
                    y_coordinates[time_step, ...],
                    z_coordinates[time_step, ...],
                )
                if draw_trajectory:
                    for i in range(n_obj):
                        line_plot = self._line_plots[i]
                        line_plot.set_xdata(x_coordinates[: time_step + 1, i])
                        line_plot.set_ydata(y_coordinates[: time_step + 1, i])
                        line_plot.set_3d_properties(
                            z_coordinates[: time_step + 1, i]
                        )

        super(NBodyPlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class SpaceLinePlot(AnimatedPlot):
    """
    An animated line plot to visualise the solutions of 1D partial differential
    equations.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        equal_scale: bool = False,
        **_,
    ):
        """
        :param y: an array representing the solution of the 1D partial
            differential equation
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param v_min: the lower y-axis limit; if None, the limit is set to the
            minimum of the solution
        :param v_max: the upper y-axis limit; if None, the limit is set to the
            maximum of the solution
        :param equal_scale: whether the scale of the values of the solution
            scalar field is the same as the scale of the spatial dimension
            (i.e. the values represent height)
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, 1, False
        )

        self._line_plot: Optional[Line2D] = None

        fig, ax = plt.subplots()

        def init_plot():
            ax.clear()
            (self._line_plot,) = ax.plot(
                mesh.coordinate_grids(vertex_oriented)[0], y[0, ..., 0]
            )

            ax.set_ylim(
                np.min(y) if v_min is None else v_min,
                np.max(y) if v_max is None else v_max,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if equal_scale:
                ax.axis("equal")

        def update_plot(time_step: int):
            self._line_plot.set_ydata(y[time_step, ..., 0])

        super(SpaceLinePlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class ContourPlot(AnimatedPlot):
    """
    A contour plot to visualize the solutions of 2D partial differential
    equations.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        color_map: Colormap = cm.viridis,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        **_,
    ):
        """
        :param y: an array representing the solution scalar field of the 2D
            partial differential equation
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param color_map: the color map to use to map the values of the
            solution scalar field to colors
        :param v_min: the lower limit of the color map; if None, the limit is
            set to the minimum of the solution
        :param v_max: the upper limit of the color map; if None, the limit is
            set to the maximum of the solution
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, 2, False
        )

        x_cartesian_coordinate_grids = mesh.cartesian_coordinate_grids(
            vertex_oriented
        )

        v_min = np.min(y) if v_min is None else v_min
        v_max = np.max(y) if v_max is None else v_max

        self._contour_plot: Optional[ContourSet] = None

        fig = plt.figure()

        def init_plot():
            fig.clear()
            ax = fig.add_subplot()
            self._contour_plot = ax.contourf(
                *x_cartesian_coordinate_grids,
                y[0, ..., 0],
                vmin=v_min,
                vmax=v_max,
                cmap=color_map,
            )

            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.axis("scaled")

            mappable = ScalarMappable(cmap=color_map)
            mappable.set_clim(v_min, v_max)
            plt.colorbar(mappable=mappable)

        def update_plot(time_step: int):
            for collection in self._contour_plot.collections:
                collection.remove()

            self._contour_plot = self._contour_plot.axes.contourf(
                *x_cartesian_coordinate_grids,
                y[time_step, ..., 0],
                vmin=v_min,
                vmax=v_max,
                cmap=color_map,
            )

        super(ContourPlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class SurfacePlot(AnimatedPlot):
    """
    A 3D surface plot to visualize the solutions of 2D partial differential
    equations.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        color_map: Colormap = cm.viridis,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        equal_scale: bool = False,
        **_,
    ):
        """
        :param y: an array representing the solution scalar field of the 2D
            partial differential equation
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param color_map: the color map to use to map the values of the
            solution scalar field to colors
        :param v_min: the lower z-axis and color map limit; if None, both of
            these limits are set to the minimum of the solution
        :param v_max: the upper z-axis and color map limit; if None, both of
            these limits are set to the maximum of the solution
        :param equal_scale: whether the scale of the values of the solution
            scalar field is the same as the scale of the spatial dimensions
            (i.e. the values represent height)
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, 2, False
        )

        x_cartesian_coordinate_grids = mesh.cartesian_coordinate_grids(
            vertex_oriented
        )

        v_min = np.min(y) if v_min is None else v_min
        v_max = np.max(y) if v_max is None else v_max

        x_0_ptp = np.ptp(x_cartesian_coordinate_grids[0])
        x_1_ptp = np.ptp(x_cartesian_coordinate_grids[1])
        x_2_ptp = (v_max - v_min) if equal_scale else min(x_0_ptp, x_1_ptp)

        surface_plot_args = {
            "vmin": v_min,
            "vmax": v_max,
            "rstride": 1,
            "cstride": 1,
            "linewidth": 0,
            "antialiased": False,
            "cmap": color_map,
        }

        self._surface_plot: Optional[Poly3DCollection] = None

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        def init_plot():
            ax.clear()
            self._surface_plot = ax.plot_surface(
                *x_cartesian_coordinate_grids,
                y[0, ..., 0],
                **surface_plot_args,
            )
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_zlabel("y")
            ax.set_zlim(v_min, v_max)
            ax.set_box_aspect((x_0_ptp, x_1_ptp, x_2_ptp))

        def update_plot(time_step: int):
            self._surface_plot.remove()
            self._surface_plot = ax.plot_surface(
                *x_cartesian_coordinate_grids,
                y[time_step, ..., 0],
                **surface_plot_args,
            )

        super(SurfacePlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class ScatterPlot(AnimatedPlot):
    """
    A 3D scatter plot to visualize the solutions of 3D partial differential
    equations.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        color_map: Colormap = cm.viridis,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        marker_shape: str = "o",
        marker_size: Union[float, np.ndarray] = 20.0,
        marker_opacity: float = 1.0,
        **_,
    ):
        """
        :param y: an array representing the solution of the 3D partial
            differential equation
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param color_map: the color map to use to map the values of the
            solution scalar field to colors
        :param v_min: the lower limit of the color map; if None, the limit is
            set to the minimum of the solution
        :param v_max: the upper limit of the color map; if None, the limit is
            set to the maximum of the solution
        :param marker_shape: the shape of the point markers
        :param marker_size: the size of the point markers
        :param marker_opacity: the opacity of the point markers
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, 3, False
        )

        x_cartesian_coordinate_grids = mesh.cartesian_coordinate_grids(
            vertex_oriented
        )

        mappable = ScalarMappable(cmap=color_map)
        mappable.set_clim(
            np.min(y) if v_min is None else v_min,
            np.max(y) if v_max is None else v_max,
        )

        self._scatter_plot: Optional[PathCollection] = None

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        def init_plot():
            ax.clear()
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_zlabel("x2")
            ax.set_box_aspect(
                (
                    np.ptp(x_cartesian_coordinate_grids[0]),
                    np.ptp(x_cartesian_coordinate_grids[1]),
                    np.ptp(x_cartesian_coordinate_grids[2]),
                )
            )
            self._scatter_plot = ax.scatter(
                *x_cartesian_coordinate_grids,
                c=mappable.to_rgba(y[0, ..., 0].flatten()),
                marker=marker_shape,
                s=marker_size,
                alpha=marker_opacity,
            )

        def update_plot(time_step: int):
            self._scatter_plot.set_color(
                mappable.to_rgba(y[time_step, ..., 0].flatten())
            )

        super(ScatterPlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class StreamPlot(AnimatedPlot):
    """
    A 2D stream plot to visualize the solution vector fields of 2D partial
    differential equation systems.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        color: str = "black",
        density: float = 1.0,
        **_,
    ):
        """
        :param y: an array representing the solution vector field of the 2D
            partial differential equation system
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param color: the color to use for the lines and arrows of the stream
            plot
        :param density: the density of the stream lines
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, 2, True
        )

        coordinate_grids = mesh.coordinate_grids(vertex_oriented)

        self._stream_plot: Optional[StreamplotSet] = None

        fig = plt.figure()

        if mesh.coordinate_system_type == CoordinateSystem.POLAR:
            (x_1_min, x_1_max), (x_0_min, x_0_max) = mesh.x_intervals
            x_1_min = 0
            x_0 = coordinate_grids[1]
            x_1 = coordinate_grids[0]
            y_0 = y[..., 1]
            y_1 = y[..., 0]

            ax = fig.add_subplot(projection="polar")

        else:
            (x_0_min, x_0_max), (x_1_min, x_1_max) = mesh.x_intervals
            x_0 = coordinate_grids[0].T
            x_1 = coordinate_grids[1].T
            y_0 = y[..., 0].transpose([0, 2, 1])
            y_1 = y[..., 1].transpose([0, 2, 1])

            ax = fig.add_subplot()

        def init_plot():
            ax.clear()
            self._stream_plot = ax.streamplot(
                x_0,
                x_1,
                y_0[0, ...],
                y_1[0, ...],
                color=color,
                density=density,
            )
            ax.set_xlim(x_0_min, x_0_max)
            ax.set_ylim(x_1_min, x_1_max)

            if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
                ax.axis("scaled")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

        def update_plot(time_step: int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.patches.clear()

            self._stream_plot.lines.remove()
            self._stream_plot = ax.streamplot(
                x_0,
                x_1,
                y_0[time_step, ...],
                y_1[time_step, ...],
                color=color,
                density=density,
            )

        super(StreamPlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )


class QuiverPlot(AnimatedPlot):
    """
    A 2D or 3D quiver plot to visualize the solution vector fields of 2D or 3D
    partial differential equation systems.
    """

    def __init__(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vertex_oriented: bool,
        n_frames: int = 100,
        interval: int = 100,
        normalize: bool = False,
        pivot: str = "middle",
        quiver_scale: float = 10.0,
        **_,
    ):
        """
        :param y: an array representing the solution vector field of the
            partial differential equation system
        :param mesh: the spatial mesh over which the solution is evaluated
        :param vertex_oriented: whether the solution is evaluated over the
            vertices or the cell centers of the mesh
        :param n_frames: the number of frames to display
        :param interval: the number of milliseconds to pause between each frame
        :param normalize: Wheter to normalize the lengths of the arrows to one
        :param pivot: the pivot point of the arrows
        :param quiver_scale: the scaling factor to apply to the arrow lengths
        :param _: any ignored extra arguments
        """
        self._verify_pde_solution_shape_matches_problem(
            y, mesh, vertex_oriented, (2, 3), True
        )

        x_cartesian_coordinate_grids = mesh.cartesian_coordinate_grids(
            vertex_oriented
        )
        unit_vector_grids = mesh.unit_vector_grids(vertex_oriented)
        y_cartesian: np.ndarray = np.asarray(
            sum(
                [
                    y[..., i : i + 1] * unit_vector_grids[i][np.newaxis, ...]
                    for i in range(mesh.dimensions)
                ]
            )
        )

        self._quiver_plot: Optional[Quiver] = None

        fig = plt.figure()

        if mesh.dimensions == 2:
            y_0 = y_cartesian[..., 0]
            y_1 = y_cartesian[..., 1]

            if normalize:
                y_magnitude = np.sqrt(np.square(y_0) + np.square(y_1))
                y_magnitude_gt_zero = y_magnitude > 0.0
                y_0[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]
                y_1[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]

            ax = fig.add_subplot()

            def init_plot():
                ax.clear()
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                self._quiver_plot = ax.quiver(
                    *x_cartesian_coordinate_grids,
                    y_0[0, ...],
                    y_1[0, ...],
                    pivot=pivot,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0 / quiver_scale,
                )
                ax.axis("scaled")

            def update_plot(time_step: int):
                self._quiver_plot.set_UVC(
                    y_0[time_step, ...], y_1[time_step, ...]
                )

        else:
            y_0 = y_cartesian[..., 0] * quiver_scale
            y_1 = y_cartesian[..., 1] * quiver_scale
            y_2 = y_cartesian[..., 2] * quiver_scale

            ax = fig.add_subplot(projection="3d")

            def init_plot():
                ax.clear()
                self._quiver_plot = ax.quiver(
                    *x_cartesian_coordinate_grids,
                    y_0[0, ...],
                    y_1[0, ...],
                    y_2[0, ...],
                    pivot=pivot,
                    normalize=normalize,
                )

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_box_aspect(
                    (
                        np.ptp(x_cartesian_coordinate_grids[0]),
                        np.ptp(x_cartesian_coordinate_grids[1]),
                        np.ptp(x_cartesian_coordinate_grids[2]),
                    )
                )

            def update_plot(time_step: int):
                self._quiver_plot.remove()
                self._quiver_plot = ax.quiver(
                    *x_cartesian_coordinate_grids,
                    y_0[time_step, ...],
                    y_1[time_step, ...],
                    y_2[time_step, ...],
                    pivot=pivot,
                    normalize=normalize,
                )

        super(QuiverPlot, self).__init__(
            fig, init_plot, update_plot, y.shape[0], n_frames, interval
        )
