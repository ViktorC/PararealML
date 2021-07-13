from typing import List

from tensorflow import Tensor, gradients, math, stack

from pararealml.core.mesh import CoordinateSystem


class Differentiator:
    """

    """

    def __init__(self):
        self._coordinate_system_type = CoordinateSystem.CARTESIAN

    @property
    def coordinate_system_type(self) -> CoordinateSystem:
        """

        @return:
        """
        return self._coordinate_system_type

    @coordinate_system_type.setter
    def coordinate_system_type(self, coordinate_system_type: CoordinateSystem):
        """

        @param coordinate_system_type:
        @return:
        """
        self._coordinate_system_type = coordinate_system_type

    def gradient(self, x: Tensor, y: Tensor, i: int, j: int) -> Tensor:
        """

        @param x:
        @param y:
        @param i:
        @param j:
        @return:
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return gradients(y[:, i:i + 1], x)[0][:, j:j + 1]
        else:
            raise ValueError

    def hessian(self, x: Tensor, y: Tensor, i: int, j: int, k: int) -> Tensor:
        """

        @param x:
        @param y:
        @param i:
        @param j:
        @param k:
        @return:
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return gradients(self.gradient(x, y, i, j), x)[0][:, k:k + 1]
        else:
            raise ValueError

    def divergence(self, x: Tensor, y: Tensor, index: List[int]) -> Tensor:
        """

        @param x:
        @param y:
        @param index:
        @return:
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return math.reduce_sum(
                stack([
                    self.gradient(x, y, index[i], i)
                    for i in range(len(x.shape[-1]) - 1)
                ]),
                axis=0)
        else:
            raise ValueError

    def curl(self, x: Tensor, y: Tensor, index: List[int]) -> Tensor:
        """

        @param x:
        @param y:
        @param index:
        @return:
        """
        x_dimension = len(x.shape[-1]) - 1
        if x_dimension == 2:
            if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
                return self.gradient(x, y, index[1], 0) - \
                       self.gradient(x, y, index[0], 1)
            else:
                raise ValueError
        elif x_dimension == 3:
            if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
                return stack(
                    [
                        self.gradient(x, y, index[2], 1) -
                        self.gradient(x, y, index[1], 2),
                        self.gradient(x, y, index[0], 2) -
                        self.gradient(x, y, index[2], 0),
                        self.gradient(x, y, index[1], 0) -
                        self.gradient(x, y, index[0], 1)
                    ],
                    axis=-1)
            else:
                raise ValueError
        else:
            raise ValueError

    def laplacian(self, x: Tensor, y: Tensor, i: int) -> Tensor:
        """

        @param x:
        @param y:
        @param i:
        @return:
        """
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return math.reduce_sum(
                stack([
                    self.hessian(x, y, i, j, j)
                    for j in range(len(x.shape[-1]) - 1)
                ]),
                axis=0)
        else:
            raise ValueError
