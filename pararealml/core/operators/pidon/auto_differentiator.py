from typing import Union

import tensorflow as tf

from pararealml.core.mesh import CoordinateSystem


class AutoDifferentiator(tf.GradientTape):
    """
    A class providing various differential operators using TensorFlow's
    auto-differentiation capabilities.
    """

    def __init__(
            self,
            persistent: bool = False,
            watch_accessed_variables: bool = True):
        """
        :param persistent: whether the gradient tape should be persistent
            allowing for the calculation of multiple differential operators
        :param watch_accessed_variables: whether to automatically watch all
            accessed variables within the context of the differentiator
        """
        super(AutoDifferentiator, self).__init__(
            persistent, watch_accessed_variables)

    def batch_gradient(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            x_axis: Union[int, tf.Tensor],
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> tf.Tensor:
        """
        Returns the element(s) of the gradient of y with respect to the element
        of x defined by x_axis.

        :param x: the input tensor
        :param y: the output tensor
        :param x_axis: the element of x to take the gradient with respect to
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the gradient of y with respect to the element of x defined by
            x_axis
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError
        if isinstance(x_axis, int):
            if not (0 <= x_axis < x.shape[-1]):
                raise ValueError
        elif isinstance(x_axis, tf.Tensor):
            if len(x_axis.shape) != 1 and x_axis.shape[0] != x.shape[0]:
                raise ValueError
        else:
            raise ValueError

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            gradient = self.batch_jacobian(y, x)
            return tf.gather(gradient, x_axis, axis=2, batch_dims=1) \
                if isinstance(x_axis, tf.Tensor) else gradient[:, :, x_axis]
        else:
            raise ValueError

    def batch_hessian(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            x_axis1: int,
            x_axis2: int,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> tf.Tensor:
        """
        Returns the element(s) of the Hessian of y with respect to the elements
        of x defined by x_axis1 and x_axis2.

        :param x: the input tensor
        :param y: the output tensor
        :param x_axis1: the first element of x to take the Hessian with respect
            to
        :param x_axis2: the second element of x to take the Hessian with
            respect to
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the Hessian of y with respect to the elements of x defined by
            x_axis1 and x_axis2
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return self.batch_gradient(
                x, self.batch_gradient(x, y, x_axis1), x_axis2)
        else:
            raise ValueError

    def batch_divergence(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> tf.Tensor:
        """
        Returns the divergence of y.

        :param x: the input tensor
        :param y: the output tensor
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the divergence of y
        """
        if x.shape != y.shape:
            raise ValueError

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack([
                    self.batch_gradient(x, y[..., i:i + 1], i)
                    for i in range(x.shape[-1])
                ]),
                axis=0)
        else:
            raise ValueError

    def batch_curl(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            curl_ind: int = 0,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> tf.Tensor:
        """
        Returns the curl_ind-th component of the curl of y.

        :param x: the input tensor
        :param y: the output tensor
        :param curl_ind: the index of the component of the curl of y to
            compute; if y is a two dimensional vector field, it must be 0
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the curl_ind-th component of the curl of y
        """
        if x.shape != y.shape:
            raise ValueError

        x_dimension = x.shape[-1]
        if x_dimension == 2:
            if curl_ind != 0:
                raise ValueError

            if coordinate_system_type == CoordinateSystem.CARTESIAN:
                return self.batch_gradient(x, y[..., 1:], 0) - \
                    self.batch_gradient(x, y[..., :1], 1)
            else:
                raise ValueError
        elif x_dimension == 3:
            if coordinate_system_type == CoordinateSystem.CARTESIAN:
                return [
                    self.batch_gradient(x, y[..., 2:], 1) -
                    self.batch_gradient(x, y[..., 1:2], 2),
                    self.batch_gradient(x, y[..., :1], 2) -
                    self.batch_gradient(x, y[..., 2:], 0),
                    self.batch_gradient(x, y[..., 1:2], 0) -
                    self.batch_gradient(x, y[..., :1], 1)
                ][curl_ind]
            else:
                raise ValueError
        else:
            raise ValueError

    def batch_laplacian(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> tf.Tensor:
        """
        Returns the Laplacian of y.

        :param x: the input tensor
        :param y: the output tensor
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the Laplacian of y
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack([
                    self.batch_hessian(x, y, i, i) for i in range(x.shape[-1])
                ]),
                axis=0)
        else:
            raise ValueError
