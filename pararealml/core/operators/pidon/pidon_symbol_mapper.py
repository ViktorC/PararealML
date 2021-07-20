from typing import Callable, Sequence

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.operators.pidon.differentiation import gradient, \
    hessian, divergence, curl, laplacian
from pararealml.core.operators.symbol_mapper import SymbolMapper


class PIDONSymbolMapper(SymbolMapper):
    """
    A symbol mapper implementation for the PIDON operator.
    """

    def __init__(self, cp: ConstrainedProblem):
        """
        :param cp: the constrained problem to create a symbol mapper for
        """
        diff_eq = cp.differential_equation

        super(PIDONSymbolMapper, self).__init__(diff_eq)

        if diff_eq.x_dimension:
            self._coordinate_system_type = cp.mesh.coordinate_system_type

    def t(self) -> Callable:
        return lambda x, y: x[:, -1:]

    def y(self, y_ind: int) -> Callable:
        return lambda x, y: y[:, y_ind:y_ind + 1]

    def y_gradient(self, y_ind: int, x_axis: int) -> Callable:
        return lambda x, y: gradient(
            x[:, :-1],
            y[:, y_ind:y_ind + 1],
            x_axis,
            self._coordinate_system_type)

    def y_hessian(self, y_ind: int, x_axis1: int, x_axis2: int) -> Callable:
        return lambda x, y: hessian(
            x[:, :-1],
            y[:, y_ind:y_ind + 1],
            x_axis1,
            x_axis2,
            self._coordinate_system_type)

    def y_divergence(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool) -> Callable:
        return lambda x, y: divergence(
            x[:, :-1],
            y[:, y_indices[0]:y_indices[-1] + 1]
            if indices_contiguous else y[:, y_indices],
            self._coordinate_system_type)

    def y_curl(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool,
            curl_ind: int) -> Callable:
        return lambda x, y: curl(
            x[:, :-1],
            y[:, y_indices[0]:y_indices[-1] + 1]
            if indices_contiguous else y[:, y_indices],
            curl_ind,
            self._coordinate_system_type)

    def y_laplacian(self, y_ind: int) -> Callable:
        return lambda x, y: laplacian(
            x[:, :-1],
            y[:, y_ind:y_ind + 1],
            self._coordinate_system_type)
