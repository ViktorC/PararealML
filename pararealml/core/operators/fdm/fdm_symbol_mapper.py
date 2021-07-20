from typing import Callable, Sequence

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.operators.fdm.differentiator import Differentiator
from pararealml.core.operators.symbol_mapper import SymbolMapper


class FDMSymbolMapper(SymbolMapper):
    """
    A symbol mapper implementation for the FDM operator.
    """

    def __init__(self, cp: ConstrainedProblem, differentiator: Differentiator):
        """
        :param cp: the constrained problem to create a symbol mapper for
        :param differentiator: the differentiator instance to use
        """
        diff_eq = cp.differential_equation

        super(FDMSymbolMapper, self).__init__(diff_eq)

        self._differentiator = differentiator

        if diff_eq.x_dimension:
            mesh = cp.mesh
            self._d_x = mesh.d_x
            self._coordinate_system_type = mesh.coordinate_system_type

    def t(self) -> Callable:
        return lambda t, y, d_y_bc_func: t

    def y(self, y_ind: int) -> Callable:
        return lambda t, y, d_y_bc_func: y[..., y_ind:y_ind + 1]

    def y_gradient(self, y_ind: int, x_axis: int) -> Callable:
        return lambda t, y, d_y_bc_func: self._differentiator.gradient(
            y[..., y_ind:y_ind + 1],
            self._d_x[x_axis],
            x_axis,
            d_y_bc_func(t)[x_axis, y_ind:y_ind + 1],
            self._coordinate_system_type)

    def y_hessian(self, y_ind: int, x_axis1: int, x_axis2: int) -> Callable:
        return lambda t, y, d_y_bc_func: self._differentiator.hessian(
            y[..., y_ind:y_ind + 1],
            self._d_x[x_axis1],
            self._d_x[x_axis2],
            x_axis1,
            x_axis2,
            d_y_bc_func(t)[x_axis1, y_ind:y_ind + 1],
            self._coordinate_system_type)

    def y_divergence(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool) -> Callable:
        if indices_contiguous:
            return lambda t, y, d_y_bc_func: self._differentiator.divergence(
                y[..., y_indices[0]:y_indices[-1] + 1],
                self._d_x,
                d_y_bc_func(t)[:, y_indices[0]:y_indices[-1] + 1],
                self._coordinate_system_type)
        else:
            return lambda t, y, d_y_bc_func: self._differentiator.divergence(
                y[..., y_indices],
                self._d_x,
                d_y_bc_func(t)[:, y_indices],
                self._coordinate_system_type)

    def y_curl(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool,
            curl_ind: int) -> Callable:
        if indices_contiguous:
            return lambda t, y, d_y_bc_func: self._differentiator.curl(
                y[..., y_indices[0]:y_indices[-1] + 1],
                self._d_x,
                curl_ind,
                d_y_bc_func(t)[:, y_indices[0]:y_indices[-1] + 1],
                self._coordinate_system_type)
        else:
            return lambda t, y, d_y_bc_func: self._differentiator.curl(
                y[..., y_indices],
                self._d_x,
                curl_ind,
                d_y_bc_func(t)[:, y_indices],
                self._coordinate_system_type)

    def y_laplacian(self, y_ind: int) -> Callable:
        return lambda t, y, d_y_bc_func: self._differentiator.laplacian(
            y[..., y_ind:y_ind + 1],
            self._d_x,
            d_y_bc_func(t)[:, y_ind:y_ind + 1],
            self._coordinate_system_type)
