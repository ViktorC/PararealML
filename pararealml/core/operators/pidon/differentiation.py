from tensorflow import Tensor, gradients, math, stack

from pararealml.core.mesh import CoordinateSystem


def gradient(
        x: Tensor,
        y: Tensor,
        x_axis: int,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN
) -> Tensor:
    """
    Returns the element(s) of the gradient of y with respect to the element of
    x defined by x_axis.

    :param x: the input tensor
    :param y: the output tensor
    :param x_axis: the element of x to take the gradient with respect to
    :param coordinate_system_type: the type of the coordinate system x is from
    :return: the gradient of y with respect to the element of x defined by
        x_axis
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError
    if not (0 <= x_axis < x.shape[-1]):
        raise ValueError

    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        return gradients(y, x)[0][:, x_axis:x_axis + 1]
    else:
        raise ValueError


def hessian(
        x: Tensor,
        y: Tensor,
        x_axis1: int,
        x_axis2: int,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN
) -> Tensor:
    """
    Returns the element(s) of the Hessian of y with respect to the elements of
    x defined by x_axis1 and x_axis2.

    :param x: the input tensor
    :param y: the output tensor
    :param x_axis1: the first element of x to take the Hessian with respect to
    :param x_axis2: the second element of x to take the Hessian with respect to
    :param coordinate_system_type: the type of the coordinate system x is from
    :return: the Hessian of y with respect to the elements of x defined by
        x_axis1 and x_axis2
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError
    if not (0 <= x_axis1 < x.shape[-1]):
        raise ValueError
    if not (0 <= x_axis2 < x.shape[-1]):
        raise ValueError

    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        return gradients(gradient(x, y, x_axis1), x)[0][:, x_axis2:x_axis2 + 1]
    else:
        raise ValueError


def divergence(
        x: Tensor,
        y: Tensor,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN
) -> Tensor:
    """
    Returns the divergence of y.

    :param x: the input tensor
    :param y: the output tensor
    :param coordinate_system_type: the type of the coordinate system x is from
    :return: the divergence of y
    """
    if x.shape != y.shape:
        raise ValueError

    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        return math.reduce_sum(
            stack([
                gradient(x, y[..., i:i + 1], i) for i in range(x.shape[-1])
            ]),
            axis=0)
    else:
        raise ValueError


def curl(
        x: Tensor,
        y: Tensor,
        curl_ind: int = 0,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN
) -> Tensor:
    """
    Returns the curl_ind-th component of the curl of y.

    :param x: the input tensor
    :param y: the output tensor
    :param curl_ind: the index of the component of the curl of y to compute; if
        y is a two dimensional vector field, it must be 0
    :param coordinate_system_type: the type of the coordinate system x is from
    :return: the curl_ind-th component of the curl of y
    """
    if x.shape != y.shape:
        raise ValueError

    x_dimension = x.shape[-1]
    if x_dimension == 2:
        if curl_ind != 0:
            raise ValueError

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return gradient(x, y[..., 1:], 0) - gradient(x, y[..., :1], 1)
        else:
            raise ValueError
    elif x_dimension == 3:
        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return [
                gradient(x, y[..., 2:], 1) - gradient(x, y[..., 1:2], 2),
                gradient(x, y[..., :1], 2) - gradient(x, y[..., 2:], 0),
                gradient(x, y[..., 1:2], 0) - gradient(x, y[..., :1], 1)
            ][curl_ind]
        else:
            raise ValueError
    else:
        raise ValueError


def laplacian(
        x: Tensor,
        y: Tensor,
        coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN
) -> Tensor:
    """
    Returns the Laplacian of y.

    :param x: the input tensor
    :param y: the output tensor
    :param coordinate_system_type: the type of the coordinate system x is from
    :return: the Laplacian of y
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError

    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        return math.reduce_sum(
            stack([hessian(x, y, i, i) for i in range(x.shape[-1])]),
            axis=0)
    else:
        raise ValueError
