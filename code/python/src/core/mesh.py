from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Tuple, Optional

import numpy as np

from fipy.meshes.abstractMesh import AbstractMesh as FiPyAbstractMesh
from fipy.meshes.uniformGrid1D import UniformGrid1D as FiPyUniformGrid1D
from fipy.meshes.uniformGrid2D import UniformGrid2D as FiPyUniformGrid2D
from fipy.meshes.uniformGrid3D import UniformGrid3D as FiPyUniformGrid3D

SpatialDomainInterval = Tuple[float, float]


class Mesh(ABC):
    """
    A mesh representing a discretised domain of arbitrary dimensionality.
    """

    @abstractmethod
    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        """
        Returns the bounds of each axis of the domain
        """

    @abstractmethod
    def d_x(self) -> Tuple[float, ...]:
        """
        Returns the step sizes along the dimensions of the domain.
        """

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the discretised domain.
        """
        raise NotImplementedError

    @abstractmethod
    def x(self, index: Tuple[int, ...]) -> Tuple[float, ...]:
        """
        Returns the coordinates of the point in the domain corresponding to the
        vertex of the mesh specified by the provided index.

        :param index: the index of a vertex of the mesh
        :return: the coordinates of the corresponding point of the domain
        """

    @abstractmethod
    def fipy_mesh(self) -> FiPyAbstractMesh:
        """
        Returns the FiPy equivalent of the mesh instance.
        """


class UniformGrid(Mesh):
    """
    A rectangular grid of arbitrary dimensionality and shape with a uniform
    spacing of grid points along each axis.
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
        self._fipy_mesh = self._create_fipy_mesh()

    def x_intervals(self) -> Tuple[SpatialDomainInterval, ...]:
        return deepcopy(self._x_intervals)

    def d_x(self) -> Tuple[float, ...]:
        return tuple(self._d_x)

    def shape(self) -> Tuple[int, ...]:
        return copy(self._shape)

    def x(self, index: Tuple[int, ...]) -> Tuple[float, ...]:
        assert len(index) == len(self._shape)
        return tuple(self._x_offset + self._d_x * index)

    def fipy_mesh(self) -> FiPyAbstractMesh:
        return self._fipy_mesh

    def _create_fipy_mesh(self) -> FiPyAbstractMesh:
        """
        Creates and returns the FiPy equivalent of the mesh.
        """
        x_dimension = len(self._x_intervals)
        if x_dimension == 1:
            mesh = FiPyUniformGrid1D(
                dx=self._d_x[0],
                nx=self._shape[0])
            mesh += np.flip(self._x_offset).reshape(1, 1)
            mesh -= (
                (self._d_x[0] / 2.,),
            )
        elif x_dimension == 2:
            mesh = FiPyUniformGrid2D(
                dx=self._d_x[1],
                dy=self._d_x[0],
                nx=self._shape[1],
                ny=self._shape[0])
            mesh += np.flip(self._x_offset).reshape(2, 1)
            mesh -= (
                (self._d_x[1] / 2.,),
                (self._d_x[0] / 2.,)
            )
        elif x_dimension == 3:
            mesh = FiPyUniformGrid3D(
                dx=self._d_x[2],
                dy=self._d_x[1],
                dz=self._d_x[0],
                nx=self._shape[2],
                ny=self._shape[1],
                nz=self._shape[0])
            mesh += np.flip(self._x_offset).reshape(3, 1)
            mesh -= (
                (self._d_x[2] / 2.,),
                (self._d_x[1] / 2.,),
                (self._d_x[0] / 2.,)
            )
        else:
            raise NotImplementedError

        return mesh

    @staticmethod
    def _calculate_shape(
            x_intervals: Tuple[SpatialDomainInterval, ...],
            d_x: Tuple[float, ...]
    ) -> Tuple[int, ...]:
        """
        Calculates the shape of the mesh.

        :param x_intervals: the bounds of each axis of the domain
        :param d_x: the step sizes to use for each axis of the domain
        :return: a tuple representing the shape of the mesh
        """
        shape = []
        for i in range(len(x_intervals)):
            x_interval = x_intervals[i]
            shape.append(round((x_interval[1] - x_interval[0]) / d_x[i] + 1))

        return tuple(shape)
