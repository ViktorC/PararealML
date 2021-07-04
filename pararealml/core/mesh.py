from copy import copy, deepcopy
from typing import Tuple, Sequence

import numpy as np

SpatialDomainInterval = Tuple[float, float]


class Mesh:
    """
    A hypercube of arbitrary dimensionality and shape with a uniform spacing of
    grid points along each axis.
    """

    def __init__(
            self,
            x_intervals: Sequence[SpatialDomainInterval],
            d_x: Sequence[float]):
        """
        :param x_intervals: the bounds of each axis of the domain
        :param d_x: the step sizes to use for each axis of the domain to create
            the mesh
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

        self._x_intervals = tuple(deepcopy(x_intervals))
        self._vertices_shape = self._calculate_shape(d_x, True)
        self._cells_shape = self._calculate_shape(d_x, False)
        self._d_x = np.array(copy(d_x))
        self._vertex_coordinates = self._calculate_coordinates(True)
        self._cell_center_coordinates = self._calculate_coordinates(False)

        self._x_vertex_offset = np.array(
            [coordinates[0] for coordinates in self._vertex_coordinates])
        self._x_cell_center_offset = np.array(
            [coordinates[0] for coordinates in self._cell_center_coordinates])

    @property
    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        """
        Returns the bounds of each axis of the domain
        """
        return deepcopy(self._x_intervals)

    @property
    def d_x(self) -> Tuple[float, ...]:
        """
        Returns the step sizes along the dimensions of the domain.
        """
        return tuple(self._d_x)

    @property
    def vertices_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the array of the vertices of the discretised
        domain.
        """
        return copy(self._vertices_shape)

    @property
    def cells_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the array of the cell centers of the discretised
        domain.
        """
        return copy(self._cells_shape)

    @property
    def vertex_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of the coordinates of the vertices of the mesh along
        each axis.
        """
        return deepcopy(self._vertex_coordinates)

    @property
    def cell_center_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of the coordinates of the cell centers of the mesh
        along each axis.
        """
        return deepcopy(self._cell_center_coordinates)

    def x(self, index: Tuple[int, ...], vertex: bool) -> Tuple[float, ...]:
        """
        Returns the coordinates of the point in the domain corresponding to the
        vertex or cell center of the mesh specified by the provided index.

        :param index: the index of a vertex or cell center of the mesh
        :param vertex: whether the point is a vertex or a cell center
        :return: the coordinates of the corresponding point of the domain
        """
        if len(index) != len(self._x_intervals):
            raise ValueError

        offset = self._x_vertex_offset if vertex \
            else self._x_cell_center_offset
        return tuple(offset + self._d_x * index)

    def shape(self, vertex_oriented: bool):
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

            coordinates.append(
                np.linspace(x_low, x_high, mesh_shape[i]))

        return tuple(coordinates)
