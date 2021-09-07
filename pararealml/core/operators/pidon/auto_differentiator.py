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
            raise ValueError(
                f'number of x instances ({x.shape[0]}) must match number of '
                f'y instances ({y.shape[0]})')
        if isinstance(x_axis, int):
            if not (0 <= x_axis < x.shape[-1]):
                raise ValueError(
                    f'x axis ({x_axis}) must be non-negative and less than '
                    f'number of x dimensions ({x.shape[-1]})')
        elif isinstance(x_axis, tf.Tensor):
            if len(x_axis.shape) != 1:
                raise ValueError('x axis must be a 1 dimensional array')
            if x_axis.shape[0] != x.shape[0]:
                raise ValueError(
                    f'length of x axis ({x_axis.shape[0]}) must match number '
                    f'of x instances ({x.shape[0]})')

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            gradient = self.batch_jacobian(y, x)
            return tf.gather(gradient, x_axis, axis=2, batch_dims=1) \
                if isinstance(x_axis, tf.Tensor) else gradient[:, :, x_axis]
        else:
            raise ValueError(
                'unsupported coordinate system type '
                f'({coordinate_system_type})')

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
            raise ValueError(
                f'number of x instances ({x.shape[0]}) must match number of '
                f'y instances ({y.shape[0]})')

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return self.batch_gradient(
                x, self.batch_gradient(x, y, x_axis1), x_axis2)
        else:
            raise ValueError(
                'unsupported coordinate system type '
                f'({coordinate_system_type})')

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
            raise ValueError(f'x shape {x.shape} must match y shape {y.shape}')

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack([
                    self.batch_gradient(x, y[..., i:i + 1], i)
                    for i in range(x.shape[-1])
                ]),
                axis=0)
        else:
            raise ValueError(
                'unsupported coordinate system type '
                f'({coordinate_system_type})')

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
            raise ValueError(f'x shape {x.shape} must match y shape {y.shape}')
        x_dimension = x.shape[-1]
        if not (2 <= x_dimension <= 3):
            raise ValueError(
                f'number of x dimensions ({x_dimension}) must be 2 or 3')
        if x_dimension == 2 and curl_ind != 0:
            raise ValueError(f'curl index ({curl_ind}) must be 0 for 2D curl')
        if not (0 <= curl_ind < x_dimension):
            raise ValueError(
                f'curl index ({curl_ind}) must be non-negative and less than '
                f'number of x dimensions ({x_dimension})')

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            if x_dimension == 2:
                return self.batch_gradient(x, y[..., 1:], 0) - \
                    self.batch_gradient(x, y[..., :1], 1)
            else:
                return [
                    self.batch_gradient(x, y[..., 2:], 1) -
                    self.batch_gradient(x, y[..., 1:2], 2),
                    self.batch_gradient(x, y[..., :1], 2) -
                    self.batch_gradient(x, y[..., 2:], 0),
                    self.batch_gradient(x, y[..., 1:2], 0) -
                    self.batch_gradient(x, y[..., :1], 1)
                ][curl_ind]
        else:
            raise ValueError(
                'unsupported coordinate system type '
                f'({coordinate_system_type})')

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
            raise ValueError(
                f'number of x instances ({x.shape[0]}) must match number of '
                f'y instances ({y.shape[0]})')

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack([
                    self.batch_hessian(x, y, i, i) for i in range(x.shape[-1])
                ]),
                axis=0)
        else:
            raise ValueError(
                'unsupported coordinate system type '
                f'({coordinate_system_type})')
