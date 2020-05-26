import numpy as np


class Differentiator:
    """
    A base class for numerical differentiators.
    """

    def gradient(self, y: np.ndarray, d_x: float) -> np.ndarray:
        """
        Returns the gradient of y with respect to x at every point of the
        mesh. If y is vector-valued, the gradient at every point is the
        Jacobian.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size used to create the mesh
        :return: the gradient of y
        """
        pass

    def divergence(self, y: np.ndarray, d_x: float) -> np.ndarray:
        """
        Returns the divergence of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size used to create the mesh
        :return: the divergence of y
        """
        pass

    def curl(self, y: np.ndarray, d_x: float) -> np.ndarray:
        """
        Returns the curl of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size used to create the mesh
        :return: the curl of y
        """
        pass

    def laplacian(self, y: np.ndarray, d_x: float) -> np.ndarray:
        """
        Returns the Laplacian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size used to create the mesh
        :return: the Laplacian of y
        """
        pass


class SimpleFiniteDifferenceMethod(Differentiator):
    """
    A numerical differentiator using simple forward, central, and backward
    finite difference.
    """

    def _gradient(
            self,
            y: np.ndarray,
            d_x: float,
            skip_diagonal: bool = False,
            skip_off_diagonal: bool = False):
        """
        Calculates the gradient with the option to omit the calculation of the
        diagonal and/or off-diagonal entries of the Jacobian (assuming it is
        a square matrix at every point of the mesh).
        """
        assert len(y.shape) > 1
        assert np.all(np.array(y.shape[:-1]) > 1)

        grad_shape = list(y.shape)
        grad_shape.append(len(y.shape) - 1)
        grad = np.empty(tuple(grad_shape))

        y_slicer = [slice(None)] * len(y.shape)
        grad_slicer = [slice(None)] * len(grad.shape)

        for y_ind in range(y.shape[-1]):
            y_slicer[-1] = y_ind
            grad_slicer[-2] = y_ind

            for axis in range(len(y.shape) - 1):
                if (skip_diagonal and axis == y_ind) or \
                        (skip_off_diagonal and axis != y_ind):
                    continue

                grad_slicer[-1] = axis

                y_slicer[axis] = 0
                y_curr = y[tuple(y_slicer)]
                y_slicer[axis] = 1
                y_next = y[tuple(y_slicer)]
                grad_slicer[axis] = 0
                y_diff = (y_next - y_curr) / d_x
                grad[tuple(grad_slicer)] = y_diff

                y_slicer[axis] = y.shape[axis] - 1
                y_curr = y[tuple(y_slicer)]
                y_slicer[axis] = y.shape[axis] - 2
                y_prev = y[tuple(y_slicer)]
                grad_slicer[axis] = y.shape[axis] - 1
                y_diff = (y_curr - y_prev) / d_x
                grad[tuple(grad_slicer)] = y_diff

                for i in range(1, y.shape[axis] - 1):
                    y_slicer[axis] = i - 1
                    y_prev = y[tuple(y_slicer)]
                    y_slicer[axis] = i + 1
                    y_next = y[tuple(y_slicer)]
                    grad_slicer[axis] = i
                    y_diff = (y_next - y_prev) / (2 * d_x)
                    grad[tuple(grad_slicer)] = y_diff

                y_slicer[axis] = slice(None)
                grad_slicer[axis] = slice(None)

            y_slicer[-1] = slice(None)
            grad_slicer[-2] = slice(None)

        return grad

    def gradient(self, y: np.ndarray, d_x: float) -> np.ndarray:
        return self._gradient(y, d_x)

    def divergence(self, y: np.ndarray, d_x: float) -> np.ndarray:
        assert len(y.shape) - 1 == y.shape[-1]

        diagonal_grad = self._gradient(y, d_x, skip_off_diagonal=True)
        div = np.diagonal(diagonal_grad.T).sum(axis=-1).T

        return div

    def curl(self, y: np.ndarray, d_x: float) -> np.ndarray:
        assert y.shape[-1] == 2 or y.shape[-1] == 3
        assert len(y.shape) - 1 == y.shape[-1]

        off_diagonal_grad = self._gradient(y, d_x, skip_diagonal=True)

        if y.shape[-1] == 2:
            curl = off_diagonal_grad[..., 1, 0] - off_diagonal_grad[..., 0, 1]
        else:
            curl = np.empty(y.shape)
            curl[..., 0] = off_diagonal_grad[..., 2, 1] - \
                off_diagonal_grad[..., 1, 2]
            curl[..., 1] = off_diagonal_grad[..., 0, 2] - \
                off_diagonal_grad[..., 2, 0]
            curl[..., 2] = off_diagonal_grad[..., 1, 0] - \
                off_diagonal_grad[..., 0, 1]

        return curl

    def laplacian(self, y: np.ndarray, d_x: float) -> np.ndarray:
        assert len(y.shape) > 1
        assert np.all(np.array(y.shape[:-1]) > 2)

        lapl = np.zeros(y.shape)
        d_x_sqr = d_x ** 2

        slicer = [slice(None)] * len(y.shape)

        for y_ind in range(y.shape[-1]):
            slicer[-1] = y_ind

            for axis in range(len(y.shape) - 1):
                slicer[axis] = 2
                y_next_next = y[tuple(slicer)]
                slicer[axis] = 1
                y_next = y[tuple(slicer)]
                slicer[axis] = 0
                y_curr = y[tuple(slicer)]
                y_2nd_diff = (y_next_next - 2 * y_next + y_curr) / d_x_sqr
                lapl[tuple(slicer)] += y_2nd_diff

                slicer[axis] = y.shape[axis] - 3
                y_prev_prev = y[tuple(slicer)]
                slicer[axis] = y.shape[axis] - 2
                y_prev = y[tuple(slicer)]
                slicer[axis] = y.shape[axis] - 1
                y_curr = y[tuple(slicer)]
                y_2nd_diff = (y_curr - 2 * y_prev + y_prev_prev) / d_x_sqr
                lapl[tuple(slicer)] += y_2nd_diff

                for i in range(1, y.shape[axis] - 1):
                    slicer[axis] = i - 1
                    y_prev = y[tuple(slicer)]
                    slicer[axis] = i + 1
                    y_next = y[tuple(slicer)]
                    slicer[axis] = i
                    y_curr = y[tuple(slicer)]
                    y_2nd_diff = (y_next - 2 * y_curr + y_prev) / d_x_sqr
                    lapl[tuple(slicer)] += y_2nd_diff

                slicer[axis] = slice(None)

            slicer[-1] = slice(None)

        return lapl
