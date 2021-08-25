from enum import Enum
from typing import Tuple, Sequence, Callable, Iterable

import numpy as np

SpatialDomainInterval = Tuple[float, float]


class CoordinateSystem(Enum):
    """
    An enumeration defining the types of coordinate systems supported.
    """
    CARTESIAN = 0,
    POLAR = 1,
    CYLINDRICAL = 2,
    SPHERICAL = 3


class Mesh:
    """
    A hyper-rectangular grid of arbitrary dimensionality and shape with a
    uniform spacing of grid points along each axis.
    """

    def __init__(
            self,
            x_intervals: Sequence[SpatialDomainInterval],
            d_x: Sequence[float],
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN):
        """
        :param x_intervals: the bounds of each axis of the domain
        :param d_x: the step sizes to use for each axis of the domain to create
            the mesh
        :param coordinate_system_type: the coordinate system type used by the
            mesh
        """
        if len(x_intervals) == 0:
            raise ValueError
        if len(x_intervals) != len(d_x):
            raise ValueError
        for interval in x_intervals:
            if len(interval) != 2:
                raise ValueError
            if interval[1] <= interval[0]:
                raise ValueError

        self._x_intervals = tuple(x_intervals)
        self._d_x = tuple(d_x)
        self._coordinate_system_type = coordinate_system_type
        self._dimensions = len(x_intervals)

        if coordinate_system_type != CoordinateSystem.CARTESIAN:
            if x_intervals[0][0] < 0:
                raise ValueError
            if x_intervals[1][1] - x_intervals[1][0] > 2 * np.pi:
                raise ValueError
            if coordinate_system_type == CoordinateSystem.POLAR:
                if self._dimensions != 2:
                    raise ValueError
            else:
                if self._dimensions != 3:
                    raise ValueError
                if coordinate_system_type == CoordinateSystem.SPHERICAL \
                        and x_intervals[2][1] - x_intervals[2][0] > np.pi:
                    raise ValueError

        self._vertices_shape = self._calculate_shape(d_x, True)
        self._cells_shape = self._calculate_shape(d_x, False)
        self._vertex_coordinates = self._calculate_coordinates(True)
        self._cell_center_coordinates = self._calculate_coordinates(False)

        self._x_vertex_offset = np.array(
            [coordinates[0] for coordinates in self._vertex_coordinates])
        self._x_cell_center_offset = np.array(
            [coordinates[0] for coordinates in self._cell_center_coordinates])

    @property
    def dimensions(self) -> int:
        """
        The number of spatial dimensions the mesh spans.
        """
        return self._dimensions

    @property
    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        """
        The bounds of each axis of the domain
        """
        return self._x_intervals

    @property
    def d_x(self) -> Tuple[float, ...]:
        """
        The step sizes along each axis of the domain.
        """
        return self._d_x

    @property
    def coordinate_system_type(self) -> CoordinateSystem:
        """
        The coordinate system type used by the mesh.
        """
        return self._coordinate_system_type

    @property
    def vertices_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array of the vertices of the discretised domain.
        """
        return self._vertices_shape

    @property
    def cells_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array of the cell centers of the discretised domain.
        """
        return self._cells_shape

    @property
    def vertex_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the vertices of the mesh along each axis.
        """
        return self._vertex_coordinates

    @property
    def cell_center_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the cell centers of the mesh along each
        axis.
        """
        return self._cell_center_coordinates

    def shape(self, vertex_oriented: bool) -> Tuple[int, ...]:
        """
        Returns the shape of the array of the discretised domain.

        :param vertex_oriented: whether the shape of the vertices or the cells
            of the mesh is to be returned
        :return: the shape of the vertices or the cells
        """
        return self.vertices_shape if vertex_oriented else self.cells_shape

    def coordinates(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of the coordinates of the vertices or cell centers
        of the mesh along each axis.

        :param vertex_oriented: whether the coordinates of the vertices or the
            cells of the mesh is to be returned
        :return: a tuple of arrays each representing the coordinates along the
            corresponding axis
        """
        return self.vertex_coordinates if vertex_oriented \
            else self.cell_center_coordinates

    def x(
            self,
            index: Tuple[int, ...],
            vertex_oriented: bool) -> Tuple[float, ...]:
        """
        Returns the coordinates of the point in the domain corresponding to the
        vertex or cell center of the mesh specified by the provided index.

        :param index: the index of a vertex or cell center of the mesh
        :param vertex_oriented: whether the point is a vertex or a cell center
        :return: the coordinates of the corresponding point of the domain
        """
        offset = self._x_vertex_offset if vertex_oriented \
            else self._x_cell_center_offset
        return tuple(offset + np.multiply(self._d_x, index))

    def all_x(self, vertex_oriented: bool) -> np.ndarray:
        """
        Returns a 2D array containing all the points of the mesh where every
        row represents the coordinates of a single point along all axes of the
        spatial domain.

        :param vertex_oriented: whether the coordinates should be those of the
            vertices or the cell centers
        :return: a 2D array of point coordinates
        """
        shape = self.shape(vertex_oriented)
        return np.stack([
            self.x(index, vertex_oriented) for index in np.ndindex(shape)
        ], axis=0)

    def evaluate_fields(
            self,
            fields: Iterable[Callable[[Sequence[float]], Sequence[float]]],
            vertex_oriented: bool,
            flatten: bool = False) -> np.ndarray:
        """
        Evaluates the provided scalar or vector fields over the mesh.

        :param fields: the field functions
        :param vertex_oriented: whether the fields are to be evaluated over the
            vertices or the cell centers of the mesh
        :param flatten: whether the field values should be flattened into a
            2D array such that every row represents a field and every column
            represents a single component of the field over a single point on
            the mesh
        :return: an 3D array where the first axis corresponds to the different
            fields, the second axis corresponds to the flattened
        """
        all_x = self.all_x(vertex_oriented)

        all_field_values = []
        for field in fields:
            field_values = []
            for i in range(all_x.shape[0]):
                sensor_point = all_x[i]
                field_values.append(field(sensor_point))

            all_field_values.append(field_values)

        all_field_values_arr = np.array(all_field_values)
        if flatten:
            all_field_values_arr = all_field_values_arr.reshape(
                (all_field_values_arr.shape[0], -1))
        return all_field_values_arr

    def _calculate_shape(
            self,
            d_x: Sequence[float],
            vertex_oriented: bool
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
            shape.append(round(
                (x_interval[1] - x_interval[0]) / d_x[i] + vertex_oriented))

        return tuple(shape)

    def _calculate_coordinates(
            self,
            vertex_oriented: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculates a tuple of the coordinates of the vertices or cell centers
        of the mesh along each axis.

        :param vertex_oriented: whether the coordinates of the vertices or the
            cell centers of the mesh are to be calculated
        :return: a tuple of arrays each representing the coordinates along the
            corresponding axis
        """
        mesh_shape = self._vertices_shape if vertex_oriented \
            else self._cells_shape

        coordinates = []
        for i, x_interval in enumerate(self._x_intervals):
            x_low = x_interval[0]
            x_high = x_interval[1]

            if not vertex_oriented:
                half_space_step = self._d_x[i] / 2.
                x_low += half_space_step
                x_high -= half_space_step

            axis_coordinates = np.linspace(x_low, x_high, mesh_shape[i])
            axis_coordinates.setflags(write=False)
            coordinates.append(axis_coordinates)

        return tuple(coordinates)


def to_cartesian_coordinates(
        x: Sequence[float],
        from_coordinate_system_type: CoordinateSystem) -> Sequence[float]:
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
            x[0] * np.cos(x[2])
        ]
    else:
        raise ValueError


def from_cartesian_coordinates(
        x: Sequence[float],
        to_coordinate_system_type: CoordinateSystem) -> Sequence[float]:
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
        return [np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan(x[1] / x[0])]
    elif to_coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        return [np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan(x[1] / x[0]), x[2]]
    elif to_coordinate_system_type == CoordinateSystem.SPHERICAL:
        return [
            np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
            np.arctan(x[1] / x[0]),
            np.arctan(np.sqrt(x[0] ** 2 + x[1] ** 2) / x[2])
        ]
    else:
        raise ValueError
