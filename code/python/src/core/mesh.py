from copy import copy, deepcopy
from typing import Tuple

import numpy as np

SpatialDomainInterval = Tuple[float, float]


class Mesh:
    """
    A mesh representing a discretised domain of arbitrary dimensionality.
    """

    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        """
        Returns the bounds of each axis of the domain
        """
        pass

    def d_x(self) -> Tuple[float, ...]:
        """
        Returns the step sizes along the dimensions of the domain.
        """
        pass

    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the discretised domain.
        """
        pass

    def x(self, index: Tuple[int, ...]) -> np.ndarray:
        """
        Returns the coordinates of the point in the domain corresponding to the
        vertex of the mesh specified by the provided index.

        :param index: the index of a vertex of the mesh
        :return: the coordinates of the corresponding point of the domain
        """
        pass


class NonUniformGrid(Mesh):
    """
    A non-uniform rectangular grid of arbitrary dimensionality and shape.
    """

    def __init__(
            self,
            x_intervals: Tuple[SpatialDomainInterval, ...],
            d_x: Tuple[float, ...]):
        """
        :param x_intervals: the bounds of each axis of the domain
        :param d_x: the step sizes to use for each axis of the domain to create
        the mesh
        """
        assert len(x_intervals) > 0
        assert len(x_intervals) == len(d_x)
        for interval in x_intervals:
            assert len(interval) == 2
            assert interval[1] > interval[0]

        self._x_intervals = deepcopy(x_intervals)
        self._shape = self._calculate_shape(x_intervals, d_x)
        self._x_offset = np.array([interval[0] for interval in x_intervals])
        self._d_x = np.array(copy(d_x))

    @staticmethod
    def _calculate_shape(
            x_intervals: Tuple[SpatialDomainInterval, ...],
            d_x: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Calculates the shape of the mesh.

        :param x_intervals: the bounds of each axis of the domain
        :param d_x: the step sizes to use for each axis of the domain
        :return: a tuple representing the shape of the mesh
        """
        shape = []
        for i in range(len(x_intervals)):
            x_interval = x_intervals[i]
            shape.append(round((x_interval[1] - x_interval[0]) / d_x[i]))

        return tuple(shape)

    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        return deepcopy(self._x_intervals)

    def d_x(self) -> Tuple[float, ...]:
        return tuple(self._d_x)

    def shape(self) -> Tuple[int, ...]:
        return copy(self._shape)

    def x(self, index: Tuple[int, ...]) -> np.ndarray:
        assert len(index) == len(self._shape)
        return self._x_offset + self._d_x * index
