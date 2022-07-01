from typing import Union

import tensorflow as tf

from pararealml.mesh import CoordinateSystem


class AutoDifferentiator(tf.GradientTape):
    """
    A class providing various differential operators using TensorFlow's
    auto-differentiation capabilities.
    """

    def __init__(
        self, persistent: bool = False, watch_accessed_variables: bool = True
    ):
        """
        :param persistent: whether the gradient tape should be persistent
            allowing for the calculation of multiple differential operators
        :param watch_accessed_variables: whether to automatically watch all
            accessed variables within the context of the differentiator
        """
        super(AutoDifferentiator, self).__init__(
            persistent, watch_accessed_variables
        )

    def batch_gradient(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        x_axis: Union[int, tf.Tensor],
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
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
        derivative = self._batch_derivative(x, y, x_axis)

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return derivative

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            if x_axis == 0:
                return derivative
            elif x_axis == 1:
                phi = x[:, 2:]
                return derivative / (r * tf.math.sin(phi))
            else:
                return derivative / r

        else:
            if x_axis == 1:
                r = x[:, :1]
                return derivative / r
            else:
                return derivative

    def batch_hessian(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        x_axis1: int,
        x_axis2: int,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
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
        second_derivative = self._batch_derivative(
            x, self._batch_derivative(x, y, x_axis1), x_axis2
        )

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return second_derivative

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            phi = x[:, 2:]

            if x_axis1 == 0 and x_axis2 == 0:
                return second_derivative

            elif x_axis1 == 1 and x_axis2 == 1:
                d_y_over_d_r = self._batch_derivative(x, y, 0)
                d_y_over_d_phi = self._batch_derivative(x, y, 2)
                return (
                    d_y_over_d_r
                    + (
                        second_derivative / tf.math.sin(phi)
                        + tf.math.cos(phi) * d_y_over_d_phi
                    )
                    / (r * tf.math.sin(phi))
                ) / r

            elif x_axis1 == 2 and x_axis2 == 2:
                d_y_over_d_r = self._batch_derivative(x, y, 0)
                return (second_derivative / r + d_y_over_d_r) / r

            elif (x_axis1 == 0 and x_axis2 == 1) or (
                x_axis1 == 1 and x_axis2 == 0
            ):
                d_y_over_d_theta = self._batch_derivative(x, y, 1)
                return (second_derivative - d_y_over_d_theta / r) / (
                    r * tf.math.sin(phi)
                )

            elif (x_axis1 == 0 and x_axis2 == 2) or (
                x_axis1 == 2 and x_axis2 == 0
            ):
                d_y_over_d_phi = self._batch_derivative(x, y, 2)
                return (second_derivative - d_y_over_d_phi / r) / r

            else:
                d_y_over_d_theta = self._batch_derivative(x, y, 1)
                return (
                    tf.math.sin(phi) * second_derivative
                    - tf.math.cos(phi) * d_y_over_d_theta
                ) / (r * tf.math.sin(phi)) ** 2

        else:
            r = x[:, :1]

            if (x_axis1 == 0 or x_axis1 == 2) and (
                x_axis2 == 0 or x_axis2 == 2
            ):
                return second_derivative

            elif x_axis1 == 1 and x_axis2 == 1:
                d_y_over_d_r = self._batch_derivative(x, y, 0)
                return (second_derivative / r + d_y_over_d_r) / r

            elif (x_axis1 == 1 and x_axis2 == 0) or (
                x_axis1 == 0 and x_axis2 == 1
            ):
                d_y_over_d_theta = self._batch_derivative(x, y, 1)
                return (second_derivative - d_y_over_d_theta / r) / r

            else:
                return second_derivative / r

    def batch_divergence(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
    ) -> tf.Tensor:
        """
        Returns the divergence of y.

        :param x: the input tensor
        :param y: the output tensor
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the divergence of y
        """
        if y.shape[1] != x.shape[1]:
            raise ValueError(
                f"number of y dimensions ({y.shape[1]}) must match number of "
                f"x dimensions ({x.shape[1]})"
            )

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack(
                    [
                        self._batch_derivative(x, y[..., i : i + 1], i)
                        for i in range(x.shape[-1])
                    ]
                ),
                axis=0,
            )

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            phi = x[:, 2:]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            y_phi = y[..., 2:]
            d_y_r_over_d_r = self._batch_derivative(x, y_r, 0)
            d_y_theta_over_d_theta = self._batch_derivative(x, y_theta, 1)
            d_y_phi_over_d_phi = self._batch_derivative(x, y_phi, 2)
            return (
                d_y_r_over_d_r
                + (
                    d_y_phi_over_d_phi
                    + tf.math.multiply(y_r, 2.0)
                    + (d_y_theta_over_d_theta + tf.math.cos(phi) * y_phi)
                    / tf.math.sin(phi)
                )
                / r
            )

        else:
            r = x[:, :1]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            d_y_r_over_d_r = self._batch_derivative(x, y_r, 0)
            d_y_theta_over_d_theta = self._batch_derivative(x, y_theta, 1)
            div = d_y_r_over_d_r + (y_r + d_y_theta_over_d_theta) / r

            if coordinate_system_type == CoordinateSystem.POLAR:
                return div
            else:
                y_z = y[..., 2:]
                d_y_z_over_d_z = self._batch_derivative(x, y_z, 2)
                return div + d_y_z_over_d_z

    def batch_curl(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        curl_ind: int = 0,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
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
        x_dimension = x.shape[1]
        if y.shape[1] != x_dimension:
            raise ValueError(
                f"number of y dimensions ({y.shape[1]}) must match number of "
                f"x dimensions ({x_dimension})"
            )
        if not (2 <= x_dimension <= 3):
            raise ValueError(
                f"number of x dimensions ({x_dimension}) must be 2 or 3"
            )
        if x_dimension == 2 and curl_ind != 0:
            raise ValueError(f"curl index ({curl_ind}) must be 0 for 2D curl")
        if not (0 <= curl_ind < x_dimension):
            raise ValueError(
                f"curl index ({curl_ind}) must be non-negative and less than "
                f"number of x dimensions ({x_dimension})"
            )

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            if x_dimension == 2 or curl_ind == 2:
                return self._batch_derivative(
                    x, y[..., 1:2], 0
                ) - self._batch_derivative(x, y[..., :1], 1)

            elif curl_ind == 0:
                return self._batch_derivative(
                    x, y[..., 2:], 1
                ) - self._batch_derivative(x, y[..., 1:2], 2)

            else:
                return self._batch_derivative(
                    x, y[..., :1], 2
                ) - self._batch_derivative(x, y[..., 2:], 0)

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            phi = x[:, 2:]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            y_phi = y[..., 2:]

            if curl_ind == 0:
                d_y_theta_over_d_phi = self._batch_derivative(x, y_theta, 2)
                d_y_phi_over_d_theta = self._batch_derivative(x, y_phi, 1)
                return (
                    d_y_theta_over_d_phi
                    + (tf.math.cos(phi) * y_theta - d_y_phi_over_d_theta)
                    / tf.math.sin(phi)
                ) / r

            elif curl_ind == 1:
                d_y_r_over_d_phi = self._batch_derivative(x, y_r, 2)
                d_y_phi_over_d_r = self._batch_derivative(x, y_phi, 0)
                return d_y_phi_over_d_r + (y_phi - d_y_r_over_d_phi) / r

            else:
                d_y_r_over_d_theta = self._batch_derivative(x, y_r, 1)
                d_y_theta_over_d_r = self._batch_derivative(x, y_theta, 0)
                return (
                    -d_y_theta_over_d_r
                    + (d_y_r_over_d_theta / tf.math.sin(phi) - y_theta) / r
                )

        else:
            r = x[:, :1]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]

            if (
                coordinate_system_type == CoordinateSystem.POLAR
                or curl_ind == 2
            ):
                d_y_r_over_d_theta = self._batch_derivative(x, y_r, 1)
                d_y_theta_over_d_r = self._batch_derivative(x, y_theta, 0)
                return d_y_theta_over_d_r + (y_theta - d_y_r_over_d_theta) / r

            elif curl_ind == 0:
                y_z = y[..., 2:]
                d_y_theta_over_d_z = self._batch_derivative(x, y_theta, 2)
                d_y_z_over_d_theta = self._batch_derivative(x, y_z, 1)
                return d_y_z_over_d_theta / r - d_y_theta_over_d_z

            else:
                y_z = y[..., 2:]
                d_y_r_over_d_z = self._batch_derivative(x, y_r, 2)
                d_y_z_over_d_r = self._batch_derivative(x, y_z, 0)
                return d_y_r_over_d_z - d_y_z_over_d_r

    def batch_laplacian(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
    ) -> tf.Tensor:
        """
        Returns the element-wise scalar Laplacian of y.

        :param x: the input tensor
        :param y: the output tensor
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the Laplacian of y
        """
        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return tf.math.reduce_sum(
                tf.stack(
                    [
                        self._batch_derivative(
                            x, self._batch_derivative(x, y, i), i
                        )
                        for i in range(x.shape[-1])
                    ]
                ),
                axis=0,
            )

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            phi = x[:, 2:]
            d_y_over_d_r = self._batch_derivative(x, y, 0)
            d_y_over_d_theta = self._batch_derivative(x, y, 1)
            d_y_over_d_phi = self._batch_derivative(x, y, 2)
            d_sqr_y_over_d_r_sqr = self._batch_derivative(x, d_y_over_d_r, 0)
            d_sqr_y_over_d_theta_sqr = self._batch_derivative(
                x, d_y_over_d_theta, 1
            )
            d_sqr_y_over_d_phi_sqr = self._batch_derivative(
                x, d_y_over_d_phi, 2
            )
            return (
                d_sqr_y_over_d_r_sqr
                + (
                    tf.math.multiply(d_y_over_d_r, 2.0)
                    + (
                        d_sqr_y_over_d_phi_sqr
                        + (
                            tf.math.cos(phi) * d_y_over_d_phi
                            + d_sqr_y_over_d_theta_sqr / tf.math.sin(phi)
                        )
                        / tf.math.sin(phi)
                    )
                    / r
                )
                / r
            )

        else:
            r = x[:, :1]
            d_y_over_d_r = self._batch_derivative(x, y, 0)
            d_y_over_d_theta = self._batch_derivative(x, y, 1)
            d_sqr_y_over_d_r_sqr = self._batch_derivative(x, d_y_over_d_r, 0)
            d_sqr_y_over_d_theta_sqr = self._batch_derivative(
                x, d_y_over_d_theta, 1
            )
            laplacian = (
                d_sqr_y_over_d_r_sqr
                + (d_sqr_y_over_d_theta_sqr / r + d_y_over_d_r) / r
            )

            if coordinate_system_type == CoordinateSystem.POLAR:
                return laplacian
            else:
                d_y_over_d_z = self._batch_derivative(x, y, 2)
                d_sqr_y_over_d_z_sqr = self._batch_derivative(
                    x, d_y_over_d_z, 2
                )
                return laplacian + d_sqr_y_over_d_z_sqr

    def batch_vector_laplacian(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        vector_laplacian_ind: int,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN,
    ) -> tf.Tensor:
        """
        Returns the vector Laplacian of y.

        :param x: the input tensor
        :param y: the output tensor
        :param vector_laplacian_ind: the index of the component of the vector
            Laplacian of y to compute
        :param coordinate_system_type: the type of the coordinate system x is
            from
        :return: the vector Laplacian of y
        """
        x_dimension = x.shape[1]
        if y.shape[1] != x_dimension:
            raise ValueError(
                f"number of y dimensions ({y.shape[1]}) must match number of "
                f"x dimensions ({x_dimension})"
            )
        if not (0 <= vector_laplacian_ind < x_dimension):
            raise ValueError(
                f"vector Laplacian index ({vector_laplacian_ind}) must be "
                "non-negative and less than number of x dimensions "
                f"({x_dimension})"
            )

        laplacian = self.batch_laplacian(
            x, y[:, vector_laplacian_ind : vector_laplacian_ind + 1]
        )

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return laplacian

        elif coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = x[:, :1]
            phi = x[:, 2:]
            y_r = y[:, :1]
            y_theta = y[:, 1:2]
            y_phi = y[:, 2:]

            if vector_laplacian_ind == 1:
                d_y_theta_over_d_theta = self._batch_derivative(x, y_theta, 1)
                d_y_phi_over_d_phi = self._batch_derivative(x, y_phi, 2)
                return (
                    laplacian
                    - tf.math.multiply(
                        y_r
                        + d_y_phi_over_d_phi
                        + (tf.cos(phi) * y_phi + d_y_theta_over_d_theta)
                        / tf.sin(phi),
                        2.0,
                    )
                    / r**2
                )

            elif vector_laplacian_ind == 2:
                d_y_r_over_d_theta = self._batch_derivative(x, y_r, 1)
                d_y_phi_over_d_theta = self._batch_derivative(x, y_phi, 1)
                return laplacian + tf.math.multiply(
                    d_y_r_over_d_theta
                    + (
                        tf.cos(phi) * d_y_phi_over_d_theta
                        - tf.math.divide(y_theta, 2.0)
                    )
                    / tf.sin(phi),
                    2.0,
                ) / (tf.sin(phi) * r**2)

            else:
                d_y_r_over_d_phi = self._batch_derivative(x, y_r, 2)
                d_y_theta_over_d_theta = self._batch_derivative(x, y_theta, 1)
                return (
                    laplacian
                    + tf.math.multiply(
                        d_y_r_over_d_phi
                        - (
                            tf.math.divide(y_phi, 2.0)
                            + tf.cos(phi) * d_y_theta_over_d_theta
                        )
                        / tf.sin(phi) ** 2,
                        2.0,
                    )
                    / r**2
                )

        else:
            r = x[:, :1]
            y_r = y[:, :1]
            y_theta = y[:, 1:2]

            if vector_laplacian_ind == 0:
                d_y_theta_over_d_theta = self._batch_derivative(x, y_theta, 1)
                return (
                    laplacian
                    - (y_r + tf.math.multiply(d_y_theta_over_d_theta, 2.0))
                    / r**2
                )

            elif vector_laplacian_ind == 1:
                d_y_r_over_d_theta = self._batch_derivative(x, y_r, 1)
                return (
                    laplacian
                    - (y_theta - tf.math.multiply(d_y_r_over_d_theta, 2.0))
                    / r**2
                )

            else:
                return laplacian

    def _batch_derivative(
        self, x: tf.Tensor, y: tf.Tensor, x_axis: Union[int, tf.Tensor]
    ) -> tf.Tensor:
        """
        Returns the element(s) of the first derivative of y with respect to the
        element of x defined by x_axis.

        :param x: the input tensor
        :param y: the output tensor
        :param x_axis: the element of x to take the gradient with respect to
        :return: the first derivative of y with respect to the element of x
            defined by x_axis
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"number of x instances ({x.shape[0]}) must match number of "
                f"y instances ({y.shape[0]})"
            )

        if isinstance(x_axis, int):
            if not (0 <= x_axis < x.shape[-1]):
                raise ValueError(
                    f"x-axis ({x_axis}) must be non-negative and less than "
                    f"number of x dimensions ({x.shape[-1]})"
                )
        elif isinstance(x_axis, tf.Tensor):
            if len(x_axis.shape) != 1:
                raise ValueError("x-axis must be a 1 dimensional array")
            if x_axis.shape[0] != x.shape[0]:
                raise ValueError(
                    f"length of x-axis ({x_axis.shape[0]}) must match number "
                    f"of x instances ({x.shape[0]})"
                )

        derivatives = self.batch_jacobian(y, x)
        return (
            tf.gather(derivatives, x_axis, axis=2, batch_dims=1)
            if isinstance(x_axis, tf.Tensor)
            else derivatives[:, :, x_axis]
        )
