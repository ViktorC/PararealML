from typing import Union, Callable, Sequence, Optional

import sympy as sp
import tensorflow as tf

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator
from pararealml.core.operators.pidon.differentiation import gradient, \
    laplacian
from pararealml.core.operators.pidon.pi_deeponet import PIDeepONet
from pararealml.core.operators.pidon.pidon_symbol_mapper import PIDONSymbolMapper
from pararealml.core.solution import Solution


class PIDONOperator(Operator):
    """
    A physics informed DeepONet based unsupervised machine learning operator
    for solving initial value problems.
    """

    def __init__(
            self,
            d_t: float,
            vertex_oriented: bool):
        """
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        """
        if d_t <= 0.:
            raise ValueError

        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._model: Optional[PIDeepONet] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

    @property
    def model(self) -> Optional[PIDeepONet]:
        """
        The DeepONet model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[PIDeepONet]):
        self._model = model

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True
    ) -> Solution:
        ...

    def train(
            self,
            ivp: InitialValueProblem,
            **training_config: Union[int, float, str]
    ):
        """
        ...

        :param ivp: the IVP to train the operator on
        :param training_config: keyworded training configuration arguments
        :return:
        """
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        symbol_set = set()
        symbolic_equation_system = diff_eq.symbolic_equation_system
        for rhs_element in symbolic_equation_system.rhs:
            symbol_set.update(rhs_element.free_symbols)

        symbol_mapper = PIDONSymbolMapper(cp)
        rhs_lambda, symbol_arg_funcs = \
            symbol_mapper.create_rhs_lambda_and_arg_functions()

        lhs_functions = self._create_lhs_functions(cp)

        def diff_eq_error(
                x: tf.Tensor,
                y: tf.Tensor
        ) -> Sequence[tf.Tensor]:
            rhs = rhs_lambda(
                [func(x, y) for func in symbol_arg_funcs]
            )
            return [
                lhs_functions[j](x, y) - rhs[j]
                for j in range(diff_eq.y_dimension)
            ]

        ...

    @staticmethod
    def _create_lhs_functions(
            cp: ConstrainedProblem
    ) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        """
        Returns a list of functions for calculating the left hand sides of the
        differential equation given x and y.

        :param cp: the constrained problem to compute the left hand sides for
        :return: a list of functions
        """
        diff_eq = cp.differential_equation
        lhs_functions = []
        for i, lhs_type in \
                enumerate(diff_eq.symbolic_equation_system.lhs_types):
            if lhs_type == Lhs.D_Y_OVER_D_T:
                lhs_functions.append(
                    lambda x, y, _i=i:
                    gradient(x[:, -1:], y[:, _i:_i + 1], 0))
            elif lhs_type == Lhs.Y:
                lhs_functions.append(lambda x, y, _i=i: y[:, _i:_i + 1])
            elif lhs_type == Lhs.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda x, y, _i=i:
                    laplacian(
                        x[:, :-1],
                        y[:, _i:_i + 1],
                        cp.mesh.coordinate_system_type))
            else:
                raise ValueError

        return lhs_functions
