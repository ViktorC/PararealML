from enum import Enum
from typing import Iterable, Sequence, Tuple, TypeVar

import numpy as np

SpatialDomainInterval = Tuple[float, float]


class CoordinateSystem(Enum):
    """
    An enumeration defining the types of coordinate systems supported.
    """

    CARTESIAN = 0
    POLAR = 1
    CYLINDRICAL = 2
    SPHERICAL = 3


class Mesh:
    """
    A hyper-rectangular grid of arbitrary dimensionality, size, and
    axis-specific uniform point spacing defined in any one of the supported
    coordinate systems. It provides both a definition and a discretisation of
    the spatial domain of any partial differential equation.
    """

    def __init__(
        self,
        x_intervals: Sequence[SpatialDomainInterval],
        d_x: Sequence[float],
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
    ):
        """
        :param x_intervals: the bounds of each axis of the spatial domain
        :param d_x: the step sizes to use for each axis of the spatial domain
            to create the mesh
        :param coordinate_system_type: the coordinate system the spatial domain
            is defined in
        """
        if len(x_intervals) == 0:
            raise ValueError(
                "number of spatial domain intervals must be greater than 0"
            )
        if len(x_intervals) != len(d_x):
            raise ValueError(
                f"number of spatial domain intervals ({len(x_intervals)}) "
                f"must match number of spatial step sizes ({len(d_x)})"
            )
        if any(interval[1] <= interval[0] for interval in x_intervals):
            raise ValueError(
                "upper bound of every spatial domain interval must be greater "
                "than its lower bound"
            )
        if any(d_x_axis <= 0.0 for d_x_axis in d_x):
            raise ValueError("all spatial step sizes must be greater than 0")

        self._x_intervals = tuple(x_intervals)
        self._d_x = tuple(d_x)
        self._coordinate_system_type = coordinate_system_type
        self._dimensions = len(x_intervals)

        if coordinate_system_type != CoordinateSystem.CARTESIAN:
            if x_intervals[0][0] < 0:
                raise ValueError(
                    f"lower bound of r interval ({x_intervals[0][0]}) "
                    "must be non-negative"
                )
            if x_intervals[1][0] < 0.0 or x_intervals[1][1] > 2.0 * np.pi:
                raise ValueError(
                    f"lower bound of theta ({x_intervals[1][0]}) must be "
                    f"non-negative and upper bound ({x_intervals[1][1]}) must "
                    f"be no more than two Pi"
                )
            if coordinate_system_type == CoordinateSystem.POLAR:
                if self._dimensions != 2:
                    raise ValueError(
                        f"number of dimensions ({self._dimensions}) of polar "
                        "mesh must be 2"
                    )
            else:
                if self._dimensions != 3:
                    raise ValueError(
                        f"number of dimensions ({self._dimensions}) of"
                        f"cylindrical and spherical meshes must be 3"
                    )
                if coordinate_system_type == CoordinateSystem.SPHERICAL and (
                    x_intervals[2][0] < 0.0 or x_intervals[2][1] > np.pi
                ):
                    raise ValueError(
                        f"lower bound of phi ({x_intervals[2][0]}) must be "
                        f"non-negative and upper bound ({x_intervals[2][1]}) "
                        f"must be no more than Pi"
                    )

        self._volume = self._compute_volume()
        self._boundary_sizes = tuple(self._compute_boundary_sizes())
        self._vertices_shape = self._create_shape(d_x, True)
        self._cells_shape = self._create_shape(d_x, False)
        self._vertex_axis_coordinates = self._create_axis_coordinates(True)
        self._cell_center_axis_coordinates = self._create_axis_coordinates(
            False
        )
        self._vertex_coordinate_grids = self._create_coordinate_grids(True)
        self._cell_center_coordinate_grids = self._create_coordinate_grids(
            False
        )

    @property
    def x_intervals(self) -> Sequence[SpatialDomainInterval]:
        """
        The bounds of each axis of the spatial domain.
        """
        return self._x_intervals

    @property
    def d_x(self) -> Sequence[float]:
        """
        The step sizes along each axis of the spatial domain.
        """
        return self._d_x

    @property
    def coordinate_system_type(self) -> CoordinateSystem:
        """
        The coordinate system the spatial domain is defined in.
        """
        return self._coordinate_system_type

    @property
    def dimensions(self) -> int:
        """
        The number of dimensions the spatial domain spans.
        """
        return self._dimensions

    @property
    def volume(self) -> float:
        """
        The volume of the spatial domain.
        """
        return self._volume

    @property
    def boundary_sizes(self) -> Sequence[Tuple[float, float]]:
        """
        The sizes of the two boundaries of the spatial domain along each axis.
        """
        return self._boundary_sizes

    @property
    def vertices_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array of the vertices of the discretized domain.
        """
        return self._vertices_shape

    @property
    def cells_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array of the cell centers of the discretized domain.
        """
        return self._cells_shape

    @property
    def vertex_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the vertices of the mesh along each axis.
        """
        return self._vertex_axis_coordinates

    @property
    def cell_center_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the cell centers of the mesh along each
        axis.
        """
        return self._cell_center_axis_coordinates

    @property
    def vertex_coordinate_grids(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of grids where each element contains the coordinates along the
        corresponding axis at all the vertices of the mesh.
        """
        return self._vertex_coordinate_grids

    @property
    def cell_center_coordinate_grids(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of grids where each element contains the coordinates along the
        corresponding axis at all the cell centers of the mesh.
        """
        return self._cell_center_coordinate_grids

    def shape(self, vertex_oriented: bool) -> Tuple[int, ...]:
        """
        Returns the shape of the array of the discretized domain.

        :param vertex_oriented: whether the shape of the vertices or the cells
            of the mesh is to be returned
        :return: the shape of the vertices or the cells
        """
        return self.vertices_shape if vertex_oriented else self.cells_shape

    def axis_coordinates(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of the coordinates of the vertices or cell centers
        of the mesh along each axis separately.

        :param vertex_oriented: whether the coordinates of the vertices or the
            cell centers of the mesh is to be returned
        :return: a tuple of arrays each representing the coordinates along the
            corresponding axis
        """
        return (
            self.vertex_axis_coordinates
            if vertex_oriented
            else self.cell_center_axis_coordinates
        )

    def coordinate_grids(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of grids where each element contains the coordinates
        along the corresponding axis at all the vertices or cell centers of the
        mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the coordinate grids
        """
        return (
            self._vertex_coordinate_grids
            if vertex_oriented
            else self._cell_center_coordinate_grids
        )

    def cartesian_coordinate_grids(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of grids where each element contains the Cartesian
        coordinates along the corresponding axis at all the vertices or cell
        centers of the mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the Cartesian coordinate grids
        """
        return tuple(
            to_cartesian_coordinates(
                self.coordinate_grids(vertex_oriented),
                self._coordinate_system_type,
            )
        )

    def all_index_coordinates(
        self, vertex_oriented: bool, flatten: bool = False
    ) -> np.ndarray:
        """
        Returns an array containing the coordinates of all the points of the
        mesh.

        :param vertex_oriented: whether the coordinates should be those of the
            vertices or the cell centers
        :param flatten: whether to flatten the array into a 2D array where each
            row represents the coordinates of a single point
        :return: an array of coordinates
        """
        coordinate_grids = self.coordinate_grids(vertex_oriented)
        index_coordinates = np.stack(coordinate_grids, axis=-1)
        if flatten:
            index_coordinates = index_coordinates.reshape(
                (-1, self._dimensions)
            )
        return index_coordinates

    def unit_vector_grids(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of orthonormal unit vector grids such that each element
        of this tuple is an array containing the Cartesian coordinates of one
        of the unit vectors of the mesh's coordinate system at each vertex or
        cell center of the mesh.

        :param vertex_oriented: whether to return the unit vectors at the
            vertices or the cell centers of the mesh
        :return: a tuple of arrays containing the unit vector grids
        """
        coordinate_grids = self.coordinate_grids(vertex_oriented)
        return tuple(
            [
                np.stack(unit_vector_grid, axis=-1)
                for unit_vector_grid in unit_vectors_at(
                    coordinate_grids, self._coordinate_system_type
                )
            ]
        )

    def _create_shape(
        self, d_x: Sequence[float], vertex_oriented: bool
    ) -> Tuple[int, ...]:
        """
        Calculates the shape of the mesh.

        :param d_x: the step sizes to use for each axis of the domain
        :param vertex_oriented: whether the shape of the vertices or the cell
            centers of the mesh are to be calculated
        :return: a tuple representing the shape of the mesh
        """
        shape = []
        for i in range(len(self._x_intervals)):
            x_interval = self._x_intervals[i]
            shape.append(
                round(
                    (x_interval[1] - x_interval[0]) / d_x[i] + vertex_oriented
                )
            )

        return tuple(shape)

    def _create_axis_coordinates(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculates a tuple of the coordinates of the vertices or cell centers
        of the mesh along each axis.

        :param vertex_oriented: whether the coordinates of the vertices or the
            cell centers of the mesh are to be calculated
        :return: a tuple of arrays each representing the coordinates along the
            corresponding axis
        """
        mesh_shape = (
            self._vertices_shape if vertex_oriented else self._cells_shape
        )

        coordinates = []
        for i, x_interval in enumerate(self._x_intervals):
            x_low = x_interval[0]
            x_high = x_interval[1]

            if not vertex_oriented:
                half_space_step = self._d_x[i] / 2.0
                x_low += half_space_step
                x_high -= half_space_step

            axis_coordinates = np.linspace(x_low, x_high, mesh_shape[i])
            axis_coordinates.setflags(write=False)
            coordinates.append(axis_coordinates)

        return tuple(coordinates)

    def _create_coordinate_grids(
        self, vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Creates a tuple of grids where each element contains the coordinates
        along the corresponding axis at all the vertices or cell centers of the
        mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the coordinate grids
        """
        coordinate_grids: Iterable[np.ndarray] = np.meshgrid(
            *self.axis_coordinates(vertex_oriented), indexing="ij"
        )
        for coordinate_grid in coordinate_grids:
            coordinate_grid.setflags(write=False)
        return tuple(coordinate_grids)

    def _compute_volume(self) -> float:
        """
        Computes the volume of the spatial domain spanned by the mesh.
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            (lower_x_bounds, upper_x_bounds) = zip(*self._x_intervals)
            return np.product(np.subtract(upper_x_bounds, lower_x_bounds))

        elif self._coordinate_system_type == CoordinateSystem.SPHERICAL:
            (r_lower, r_upper) = self._x_intervals[0]
            (theta_lower, theta_upper) = self._x_intervals[1]
            (phi_lower, phi_upper) = self._x_intervals[2]
            return (
                (r_upper**3 - r_lower**3)
                / 3.0
                * (theta_upper - theta_lower)
                * (np.cos(phi_lower) - np.cos(phi_upper))
            )

        else:
            (r_lower, r_upper) = self._x_intervals[0]
            (theta_lower, theta_upper) = self._x_intervals[1]
            base_area = (
                (r_upper**2 - r_lower**2)
                * (theta_upper - theta_lower)
                / 2.0
            )

            if self._dimensions == 2:
                return base_area

            (z_lower, z_upper) = self._x_intervals[2]
            return base_area * (z_upper - z_lower)

    def _compute_boundary_sizes(self) -> Sequence[Tuple[float, float]]:
        """
        Computes the sizes of the two boundaries of the spatial mesh along each
        axis.
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            (lower_x_bounds, upper_x_bounds) = zip(*self._x_intervals)
            x_interval_lengths = np.subtract(upper_x_bounds, lower_x_bounds)
            volume = np.product(x_interval_lengths)
            return [
                (volume / x_interval_length,) * 2
                for x_interval_length in x_interval_lengths
            ]

        elif self._coordinate_system_type == CoordinateSystem.SPHERICAL:
            (r_lower, r_upper) = self._x_intervals[0]
            (phi_lower, phi_upper) = self._x_intervals[2]
            theta_span = self._x_intervals[1][1] - self._x_intervals[1][0]

            r_axis_boundary_sizes = (
                r_lower**2
                * theta_span
                * (np.cos(phi_lower) - np.cos(phi_upper)),
                r_upper**2
                * theta_span
                * (np.cos(phi_lower) - np.cos(phi_upper)),
            )
            theta_axis_boundary_sizes = (
                (r_upper**2 - r_lower**2) / 2.0 * (phi_upper - phi_lower),
            ) * 2
            phi_axis_boundary_sizes = (
                (r_upper**2 - r_lower**2)
                / 2.0
                * theta_span
                * np.sin(phi_lower),
                (r_upper**2 - r_lower**2)
                / 2.0
                * theta_span
                * np.sin(phi_upper),
            )
            return [
                r_axis_boundary_sizes,
                theta_axis_boundary_sizes,
                phi_axis_boundary_sizes,
            ]

        else:
            (r_lower, r_upper) = self._x_intervals[0]
            theta_span = self._x_intervals[1][1] - self._x_intervals[1][0]

            r_axis_boundary_sizes = (
                r_lower * theta_span,
                r_upper * theta_span,
            )
            theta_axis_boundary_sizes = ((r_upper - r_lower),) * 2

            if self._dimensions == 2:
                return [r_axis_boundary_sizes, theta_axis_boundary_sizes]

            z_span = self._x_intervals[2][1] - self._x_intervals[2][0]

            r_axis_boundary_sizes = (
                r_axis_boundary_sizes[0] * z_span,
                r_axis_boundary_sizes[1] * z_span,
            )
            theta_axis_boundary_sizes = (
                theta_axis_boundary_sizes[0] * z_span,
                theta_axis_boundary_sizes[1] * z_span,
            )
            z_axis_boundary_sizes = (
                (r_upper**2 - r_lower**2) * theta_span / 2.0,
            ) * 2
            return [
                r_axis_boundary_sizes,
                theta_axis_boundary_sizes,
                z_axis_boundary_sizes,
            ]


Coordinate = TypeVar("Coordinate", float, np.ndarray)
Coordinates = Sequence[Coordinate]


def unit_vectors_at(
    x: Coordinates, coordinate_system_type: CoordinateSystem
) -> Sequence[Coordinates]:
    """
    Calculates the unit vectors of the specified coordinate system at the
    provided spatial coordinates.

    Each element of the returned sequence represents one unit vector in
    Cartesian coordinates.

    :param x: the spatial coordinates
    :param coordinate_system_type: the coordinate system to compute the unit
        vectors in
    :return: the sequence of unit vectors at the provided spatial coordinates
    """
    unit_vectors = []

    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        for i in range(len(x)):
            zero = np.zeros_like(x[i])
            one = np.ones_like(x[i])
            unit_vector = [zero] * len(x)
            unit_vector[i] = one
            unit_vectors.append(unit_vector)

    elif coordinate_system_type == CoordinateSystem.POLAR:
        theta = x[1]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        unit_vectors.append([cos_theta, sin_theta])
        unit_vectors.append([-sin_theta, cos_theta])

    elif coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        theta = x[1]
        zero = np.zeros_like(theta)
        one = np.ones_like(theta)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        unit_vectors.append([cos_theta, sin_theta, zero])
        unit_vectors.append([-sin_theta, cos_theta, zero])
        unit_vectors.append([zero, zero, one])

    elif coordinate_system_type == CoordinateSystem.SPHERICAL:
        theta, phi = x[1], x[2]
        zero = np.zeros_like(theta)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        unit_vectors.append(
            [sin_phi * cos_theta, sin_phi * sin_theta, cos_phi]
        )
        unit_vectors.append([-sin_theta, cos_theta, zero])
        unit_vectors.append(
            [cos_phi * cos_theta, cos_phi * sin_theta, -sin_phi]
        )

    else:
        raise ValueError(
            "unsupported coordinate system type "
            f"({coordinate_system_type.name})"
        )

    return unit_vectors


def to_cartesian_coordinates(
    x: Coordinates, from_coordinate_system_type: CoordinateSystem
) -> Coordinates:
    """
    Converts the provided coordinates from the specified type of coordinate
    system to Cartesian coordinates.

    :param x: the coordinates to convert to Cartesian coordinates
    :param from_coordinate_system_type: the coordinate system the coordinates
        are from
    :return: the coordinates converted to Cartesian coordinates
    """
    if from_coordinate_system_type == CoordinateSystem.CARTESIAN:
        return x

    elif from_coordinate_system_type == CoordinateSystem.POLAR:
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]

    elif from_coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1]), x[2]]

    elif from_coordinate_system_type == CoordinateSystem.SPHERICAL:
        return [
            x[0] * np.sin(x[2]) * np.cos(x[1]),
            x[0] * np.sin(x[2]) * np.sin(x[1]),
            x[0] * np.cos(x[2]),
        ]

    else:
        raise ValueError(
            "unsupported coordinate system type "
            f"({from_coordinate_system_type.name})"
        )


def from_cartesian_coordinates(
    x: Coordinates, to_coordinate_system_type: CoordinateSystem
) -> Coordinates:
    """
    Converts the provided Cartesian coordinates to the specified type of
    coordinate system.

    :param x: the Cartesian coordinates to convert
    :param to_coordinate_system_type: the coordinate system to convert the
        Cartesian coordinates to
    :return: the converted coordinates
    """
    if to_coordinate_system_type == CoordinateSystem.CARTESIAN:
        return x

    elif to_coordinate_system_type == CoordinateSystem.POLAR:
        return [np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])]

    elif to_coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        return [np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0]), x[2]]

    elif to_coordinate_system_type == CoordinateSystem.SPHERICAL:
        return [
            np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
            np.arctan2(x[1], x[0]),
            np.arctan2(np.sqrt(x[0] ** 2 + x[1] ** 2), x[2]),
        ]

    else:
        raise ValueError(
            "unsupported coordinate system type "
            f"({to_coordinate_system_type.name})"
        )
