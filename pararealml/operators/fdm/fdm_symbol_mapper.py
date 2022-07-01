from typing import Callable, NamedTuple, Sequence, Union

import numpy as np

from pararealml import LHS
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.operators.fdm.numerical_differentiator import (
    NumericalDifferentiator,
)
from pararealml.operators.symbol_mapper import SymbolMapper


class FDMSymbolMapArg(NamedTuple):
    """
    The arguments to the FDM map functions.
    """

    t: float
    y: np.ndarray
    d_y_constraint_function: Callable[[float], np.ndarray]


FDMSymbolMapFunction = Callable[[FDMSymbolMapArg], np.ndarray]


class FDMSymbolMapper(SymbolMapper[FDMSymbolMapArg, np.ndarray]):
    """
    A symbol mapper implementation for the FDM operator.
    """

    def __init__(
        self, cp: ConstrainedProblem, differentiator: NumericalDifferentiator
    ):
        """
        :param cp: the constrained problem to create a symbol mapper for
        :param differentiator: the numerical differentiator instance to use
        """
        diff_eq = cp.differential_equation

        super(FDMSymbolMapper, self).__init__(diff_eq)

        self._differentiator = differentiator
        self._mesh = cp.mesh

    def t_map_function(self) -> FDMSymbolMapFunction:
        return lambda arg: np.array([arg.t])

    def y_map_function(self, y_ind: int) -> FDMSymbolMapFunction:
        return lambda arg: arg.y[..., y_ind : y_ind + 1]

    def x_map_function(self, x_axis: int) -> FDMSymbolMapFunction:
        return lambda arg: self._mesh.vertex_coordinate_grids[x_axis][
            ..., np.newaxis
        ]

    def y_gradient_map_function(
        self, y_ind: int, x_axis: int
    ) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.gradient(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            x_axis,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_hessian_map_function(
        self, y_ind: int, x_axis1: int, x_axis2: int
    ) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.hessian(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            x_axis1,
            x_axis2,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_divergence_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
    ) -> FDMSymbolMapFunction:
        if indices_contiguous:
            return lambda arg: self._differentiator.divergence(
                arg.y[..., y_indices[0] : y_indices[-1] + 1],
                self._mesh,
                arg.d_y_constraint_function(arg.t)[
                    :, y_indices[0] : y_indices[-1] + 1
                ],
            )
        else:
            return lambda arg: self._differentiator.divergence(
                arg.y[..., y_indices],
                self._mesh,
                arg.d_y_constraint_function(arg.t)[:, y_indices],
            )

    def y_curl_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        curl_ind: int,
    ) -> FDMSymbolMapFunction:
        if indices_contiguous:
            return lambda arg: self._differentiator.curl(
                arg.y[..., y_indices[0] : y_indices[-1] + 1],
                self._mesh,
                curl_ind,
                arg.d_y_constraint_function(arg.t)[
                    :, y_indices[0] : y_indices[-1] + 1
                ],
            )
        else:
            return lambda arg: self._differentiator.curl(
                arg.y[..., y_indices],
                self._mesh,
                curl_ind,
                arg.d_y_constraint_function(arg.t)[:, y_indices],
            )

    def y_laplacian_map_function(self, y_ind: int) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.laplacian(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_vector_laplacian_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        vector_laplacian_ind: int,
    ) -> FDMSymbolMapFunction:
        if indices_contiguous:
            return lambda arg: self._differentiator.vector_laplacian(
                arg.y[..., y_indices[0] : y_indices[-1] + 1],
                self._mesh,
                vector_laplacian_ind,
                arg.d_y_constraint_function(arg.t)[
                    :, y_indices[0] : y_indices[-1] + 1
                ],
            )
        else:
            return lambda arg: self._differentiator.vector_laplacian(
                arg.y[..., y_indices],
                self._mesh,
                vector_laplacian_ind,
                arg.d_y_constraint_function(arg.t)[:, y_indices],
            )

    def map_concatenated(
        self, arg: FDMSymbolMapArg, lhs_type: LHS
    ) -> np.ndarray:
        """
        Evaluates the right-hand side of the differential equation system
        given the map argument and concatenates the resulting sequence of map
        value arrays along the last axis.
        """
        return np.concatenate(self.map(arg, lhs_type), axis=-1)
