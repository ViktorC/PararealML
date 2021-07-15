from tensorflow import Tensor, gradients, math, stack

from pararealml.core.mesh import CoordinateSystem


class Differentiator:
    """
    A differentiator providing various differential operators for TensorFlow
    tensors.
    """

    def __init__(self):
        self._coordinate_system_type = CoordinateSystem.CARTESIAN

    @property
    def coordinate_system_type(self) -> CoordinateSystem:
        """
        Returns the coordinate system type of the differentiator.
        """
        return self._coordinate_system_type

    @coordinate_system_type.setter
    def coordinate_system_type(self, coordinate_system_type: CoordinateSystem):
        """
        Sets the coordinate system type of the differentiator.
        """
        self._coordinate_system_type = coordinate_system_type

    def gradient(self, x: Tensor, y: Tensor, x_axis: int) -> Tensor:
        """

        :param x:
        :param y:
        :param x_axis:
        :return:
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError
        if not (0 <= x_axis < x.shape[-1]):
            raise ValueError

        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return gradients(y, x)[0][:, x_axis:x_axis + 1]
        else:
            raise ValueError

    def hessian(
            self, x: Tensor, y: Tensor, x_axis1: int, x_axis2: int) -> Tensor:
        """

        :param x:
        :param y:
        :param x_axis1:
        :param x_axis2:
        :return:
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError
        if not (0 <= x_axis1 < x.shape[-1]):
            raise ValueError
        if not (0 <= x_axis2 < x.shape[-1]):
            raise ValueError

        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return gradients(
                self.gradient(x, y, x_axis1), x
            )[0][:, x_axis2:x_axis2 + 1]
        else:
            raise ValueError

    def divergence(self, x: Tensor, y: Tensor) -> Tensor:
        """

        :param x:
        :param y:
        :return:
        """
        if x.shape != y.shape:
            raise ValueError

        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return math.reduce_sum(
                stack([
                    self.gradient(x, y[..., i:i + 1], i)
                    for i in range(len(x.shape[-1]) - 1)
                ]),
                axis=0)
        else:
            raise ValueError

    def curl(self, x: Tensor, y: Tensor, curl_ind: int = 0) -> Tensor:
        """

        :param x:
        :param y:
        :param curl_ind:
        :return:
        """
        if x.shape != y.shape:
            raise ValueError

        x_dimension = x.shape[-1]
        if x_dimension == 2:
            if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
                return self.gradient(x, y[..., 1:], 0) - \
                       self.gradient(x, y[..., :1], 1)
            else:
                raise ValueError
        elif x_dimension == 3:
            if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
                return [
                    self.gradient(x, y[..., 2:], 1) -
                    self.gradient(x, y[..., 1:2], 2),
                    self.gradient(x, y[..., :1], 2) -
                    self.gradient(x, y[..., 2:], 0),
                    self.gradient(x, y[..., 1:2], 0) -
                    self.gradient(x, y[..., :1], 1)
                ][curl_ind]
            else:
                raise ValueError
        else:
            raise ValueError

    def laplacian(self, x: Tensor, y: Tensor) -> Tensor:
        """

        :param x:
        :param y:
        :return:
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError

        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            return math.reduce_sum(
                stack([self.hessian(x, y, i, i) for i in range(x.shape[-1])]),
                axis=0)
        else:
            raise ValueError
