from enum import Enum
from typing import Tuple, Sequence, Callable, Iterable, TypeVar

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

        self._vertices_shape = self._create_shape(d_x, True)
        self._cells_shape = self._create_shape(d_x, False)
        self._vertex_coordinates = self._create_axis_coordinates(True)
        self._cell_center_coordinates = self._create_axis_coordinates(False)
        self._vertex_coordinate_grids = self._create_coordinate_grids(True)
        self._cell_center_coordinate_grids = \
            self._create_coordinate_grids(False)

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
    def vertex_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the vertices of the mesh along each axis.
        """
        return self._vertex_coordinates

    @property
    def cell_center_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        A tuple of the coordinates of the cell centers of the mesh along each
        axis.
        """
        return self._cell_center_coordinates

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
        Returns the shape of the array of the discretised domain.

        :param vertex_oriented: whether the shape of the vertices or the cells
            of the mesh is to be returned
        :return: the shape of the vertices or the cells
        """
        return self.vertices_shape if vertex_oriented else self.cells_shape

    def axis_coordinates(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of the coordinates of the vertices or cell centers
        of the mesh along each axis separately.

        :param vertex_oriented: whether the coordinates of the vertices or the
            cell centers of the mesh is to be returned
        :return: a tuple of arrays each representing the coordinates along the
            corresponding axis
        """
        return self.vertex_axis_coordinates if vertex_oriented \
            else self.cell_center_axis_coordinates

    def coordinate_grids(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of grids where each element contains the coordinates
        along the corresponding axis at all the vertices or cell centers of the
        mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the coordinate grids
        """
        return self._vertex_coordinate_grids if vertex_oriented \
            else self._cell_center_coordinate_grids

    def cartesian_coordinate_grids(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of grids where each element contains the Cartesian
        coordinates along the corresponding axis at all the vertices or cell
        centers of the mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the Cartesian coordinate grids
        """
        return tuple(to_cartesian_coordinates(
            self.coordinate_grids(vertex_oriented),
            self._coordinate_system_type))

    def all_index_coordinates(
            self,
            vertex_oriented: bool,
            flatten: bool = False) -> np.ndarray:
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
            index_coordinates = \
                index_coordinates.reshape((-1, self._dimensions))
        return index_coordinates

    def unit_vector_grids(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of unit vector grids such that each element of this
        tuple is an array containing the Cartesian coordinates of one of the
        unit vectors of the mesh's coordinate system at each vertex or cell
        center of the mesh.

        :param vertex_oriented: whether to return the unit vectors at the
            vertices or the cell centers of the mesh
        :return: a tuple of arrays containing the unit vector grids
        """
        unit_vector_grids = []

        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            for i in range(self._dimensions):
                unit_vector_grid = np.zeros(
                    self.shape(vertex_oriented) + (self._dimensions,))
                unit_vector_grid[..., i] = 1.
                unit_vector_grids.append(unit_vector_grid)

        elif self._coordinate_system_type == CoordinateSystem.POLAR:
            r, theta = self.coordinate_grids(vertex_oriented)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            unit_vector_grids.append(
                np.stack((cos_theta, sin_theta), axis=-1))
            unit_vector_grids.append(
                np.stack((-sin_theta, cos_theta), axis=-1))

        elif self._coordinate_system_type == CoordinateSystem.CYLINDRICAL:
            r, theta, z = self.coordinate_grids(vertex_oriented)
            zero = np.zeros_like(z)
            one = np.ones_like(z)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            unit_vector_grids.append(
                np.stack((cos_theta, sin_theta, zero), axis=-1))
            unit_vector_grids.append(
                np.stack((-sin_theta, cos_theta, zero), axis=-1))
            unit_vector_grids.append(np.stack((zero, zero, one), axis=-1))

        elif self._coordinate_system_type == CoordinateSystem.SPHERICAL:
            r, theta, phi = self.coordinate_grids(vertex_oriented)
            zero = np.zeros_like(r)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            unit_vector_grids.append(
                np.stack(
                    (sin_phi * cos_theta, sin_phi * sin_theta, cos_phi),
                    axis=-1))
            unit_vector_grids.append(
                np.stack(
                    (cos_phi * cos_theta, cos_phi * sin_theta, -sin_phi),
                    axis=-1))
            unit_vector_grids.append(
                np.stack((-sin_theta, cos_theta, zero), axis=-1))

        else:
            raise ValueError

        return tuple(unit_vector_grids)

    def evaluate(
            self,
            functions: Iterable[Callable[[Sequence[float]], Sequence[float]]],
            vertex_oriented: bool,
            flatten: bool = False) -> np.ndarray:
        """
        Evaluates the provided vector functions over the mesh.

        :param functions: the vector functions of the spatial coordinates; all
            these functions should output sequences of the same length
        :param vertex_oriented: whether the functions are to be evaluated over
            the vertices or the cell centers of the mesh
        :param flatten: whether to flatten the evaluated values into a 2D array
            where the first axis corresponds to the different functions and the
            second axis corresponds to the values of the function evaluated
            over the vertices or cell centers of the mesh
        :return: an array containing the values of the vector functions over
            the vertices or cell centers of the mesh
        """
        all_x = self.all_index_coordinates(vertex_oriented, flatten=True)
        all_values = []
        for function in functions:
            values = [function(all_x[i]) for i in range(all_x.shape[0])]
            all_values.append(values)

        if flatten:
            return np.array(all_values).reshape((len(all_values), -1))

        return np.array(all_values).reshape(
            (len(all_values),) + self.shape(vertex_oriented) + (-1,))

    def _create_shape(
            self,
            d_x: Sequence[float],
            vertex_oriented: bool) -> Tuple[int, ...]:
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

    def _create_axis_coordinates(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
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

    def _create_coordinate_grids(
            self,
            vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        """
        Creates a tuple of grids where each element contains the coordinates
        along the corresponding axis at all the vertices or cell centers of the
        mesh.

        :param vertex_oriented: whether to return grids of coordinates at the
            vertices or the cell centers of the mesh
        :return: a tuple arrays containing the coordinate grids
        """
        coordinate_grids: Iterable[np.ndarray] = np.meshgrid(
            *self.axis_coordinates(vertex_oriented), indexing='ij')
        for coordinate_grid in coordinate_grids:
            coordinate_grid.setflags(write=False)
        return tuple(coordinate_grids)


Coordinates = TypeVar('Coordinates', Sequence[float], Sequence[np.ndarray])


def to_cartesian_coordinates(
        x: Coordinates,
        from_coordinate_system_type: CoordinateSystem) -> Coordinates:
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
        x: Coordinates,
        to_coordinate_system_type: CoordinateSystem) -> Coordinates:
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
            np.arctan2(np.sqrt(x[0] ** 2 + x[1] ** 2), x[2])
        ]
    else:
        raise ValueError
