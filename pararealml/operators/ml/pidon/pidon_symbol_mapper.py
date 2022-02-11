from typing import Callable, Sequence, NamedTuple, Optional, Union

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.operators.ml.pidon.auto_differentiator import \
    AutoDifferentiator
from pararealml.operators.symbol_mapper import SymbolMapper


class PIDONSymbolMapArg(NamedTuple):
    """
    The arguments to the PIDON map functions.
    """
    auto_diff: AutoDifferentiator
    t: tf.Tensor
    x: Optional[tf.Tensor]
    y_hat: tf.Tensor


PIDONSymbolMapFunction = Callable[[PIDONSymbolMapArg], tf.Tensor]


class PIDONSymbolMapper(SymbolMapper[PIDONSymbolMapArg, tf.Tensor]):
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
        else:
            self._coordinate_system_type = None

    def t_map_function(self) -> PIDONSymbolMapFunction:
        return lambda arg: arg.t

    def y_map_function(self, y_ind: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.y_hat[:, y_ind:y_ind + 1]

    def x_map_function(self, x_axis: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.x[:, x_axis:x_axis + 1]

    def y_gradient_map_function(self, y_ind: int, x_axis: int) -> Callable:
        return lambda arg: arg.auto_diff.batch_gradient(
            arg.x,
            arg.y_hat[:, y_ind:y_ind + 1],
            x_axis,
            self._coordinate_system_type)

    def y_hessian_map_function(
            self,
            y_ind: int,
            x_axis1: int,
            x_axis2: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_hessian(
            arg.x,
            arg.y_hat[:, y_ind:y_ind + 1],
            x_axis1,
            x_axis2,
            self._coordinate_system_type)

    def y_divergence_map_function(
            self,
            y_indices: Sequence[int],
            indices_contiguous: Union[bool, np.bool_]
    ) -> PIDONSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_divergence(
            arg.x,
            arg.y_hat[:, y_indices[0]:y_indices[-1] + 1]
            if indices_contiguous else arg.y_hat[:, y_indices],
            self._coordinate_system_type)

    def y_curl_map_function(
            self,
            y_indices: Sequence[int],
            indices_contiguous: Union[bool, np.bool_],
            curl_ind: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_curl(
            arg.x,
            arg.y_hat[:, y_indices[0]:y_indices[-1] + 1]
            if indices_contiguous else arg.y_hat[:, y_indices],
            curl_ind,
            self._coordinate_system_type)

    def y_laplacian_map_function(
            self,
            y_ind: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_laplacian(
            arg.x,
            arg.y_hat[:, y_ind:y_ind + 1],
            self._coordinate_system_type)

    def y_vector_laplacian_map_function(
            self,
            y_indices: Sequence[int],
            indices_contiguous: Union[bool, np.bool_],
            vector_laplacian_ind: int) -> PIDONSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_vector_laplacian(
            arg.x,
            arg.y_hat[:, y_indices[0]:y_indices[-1] + 1]
            if indices_contiguous else arg.y_hat[:, y_indices],
            vector_laplacian_ind,
            self._coordinate_system_type)
