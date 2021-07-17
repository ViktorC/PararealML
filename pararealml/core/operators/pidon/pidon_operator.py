from typing import Union, Callable, Sequence, Dict, Optional

import numpy as np
import sympy as sp
import tensorflow as tf

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs, DifferentialEquation
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator
from pararealml.core.operators.pidon.deeponet import DeepONet
from pararealml.core.operators.pidon.differentiation import gradient, \
    hessian, divergence, curl, laplacian
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
        self._model: Optional[DeepONet] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

    @property
    def model(self) -> Optional[DeepONet]:
        """
        The DeepONet model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[DeepONet]):
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

        symbol_map = self._create_symbol_map(ivp.constrained_problem)
        symbol_arg_funcs = [symbol_map[symbol] for symbol in symbol_set]

        rhs_lambda = sp.lambdify(
            [symbol_set],
            symbolic_equation_system.rhs,
            'numpy')

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

    @staticmethod
    def _create_symbol_map(
            cp: ConstrainedProblem
    ) -> Dict[sp.Symbol, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        """
        Creates a dictionary mapping symbols to functions returning the values
        of these symbols given x and y.

        :param cp: the constrained problem to create a symbol map for
        :return: a dictionary mapping symbols to functions
        """
        diff_eq = cp.differential_equation

        symbol_map = {diff_eq.symbols.t: lambda x, y: x[:, -1:]}

        for i, y_element in enumerate(diff_eq.symbols.y):
            symbol_map[y_element] = lambda x, y, _i=i: y[:, _i:_i + 1]

        if diff_eq.x_dimension:
            coordinate_system_type = cp.mesh.coordinate_system_type

            y_gradient = diff_eq.symbols.y_gradient
            y_hessian = diff_eq.symbols.y_hessian
            y_laplacian = diff_eq.symbols.y_laplacian
            y_divergence = diff_eq.symbols.y_divergence
            y_curl = diff_eq.symbols.y_curl

            for i in range(diff_eq.y_dimension):
                symbol_map[y_laplacian[i]] = \
                    lambda x, y, _i=i: \
                    laplacian(
                        x[:, :-1],
                        y[:, _i:_i + 1],
                        coordinate_system_type)

                for j in range(diff_eq.x_dimension):
                    symbol_map[y_gradient[i, j]] = \
                        lambda x, y, _i=i, _j=j: \
                        gradient(
                            x[:, :-1],
                            y[:, _i:_i + 1],
                            _j,
                            coordinate_system_type)

                    for k in range(diff_eq.x_dimension):
                        symbol_map[y_hessian[i, j, k]] = \
                            lambda x, y, _i=i, _j=j, _k=k: \
                            hessian(
                                x[:, :-1],
                                y[:, _i:_i + 1],
                                _j,
                                _k,
                                coordinate_system_type)

            for index in np.ndindex(
                    (diff_eq.y_dimension,) * diff_eq.x_dimension):
                symbol_map[y_divergence[index]] = lambda x, y, _index=index: \
                    divergence(x[:, :-1], y[:, _index], coordinate_system_type)
                if diff_eq.x_dimension == 2:
                    symbol_map[y_curl[index]] = \
                        lambda x, y, _index=index: \
                        curl(
                            x[:, :-1],
                            y[:, _index],
                            coordinate_system_type=coordinate_system_type)
                elif diff_eq.x_dimension == 3:
                    for curl_ind in range(3):
                        symbol_map[y_curl[index + (curl_ind,)]] = \
                            lambda x, y, _index=index, _curl_ind=curl_ind: \
                            curl(
                                x[:, :-1],
                                y[:, _index],
                                _curl_ind,
                                coordinate_system_type)

        return symbol_map
