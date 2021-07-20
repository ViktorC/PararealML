from abc import ABC, abstractmethod
from typing import Callable, Sequence, Dict

import numpy as np
import sympy as sp

from pararealml import Lhs
from pararealml.core.differential_equation import DifferentialEquation


class SymbolMapper(ABC):
    """
    A class for mapping the symbols of differential equation to functions
    for replacing the symbols with numerical values.
    """

    def __init__(self, diff_eq: DifferentialEquation):
        """
        :param diff_eq: the differential equation to create a symbol mapper for
        """
        self._diff_eq = diff_eq

    @abstractmethod
    def t(self) -> Callable:
        """
        Returns a function for substituting t for a numerical value.
        """

    @abstractmethod
    def y(self, y_ind: int) -> Callable:
        """
        Returns a function for substituting a component of y for a numerical
        value.

        :param y_ind: the component of y to substitute for
        :return: the substitution function for y
        """

    @abstractmethod
    def y_gradient(self, y_ind: int, x_axis: int) -> Callable:
        """
        Returns a function for substituting a component of the gradient of y
        for a numerical value.

        :param y_ind: the component of y whose gradient to substitute for
        :param x_axis: the x axis denoting the element of the gradient to
            substitute for
        :return: the substitution function for the gradient of y
        """

    @abstractmethod
    def y_hessian(self, y_ind: int, x_axis1: int, x_axis2: int) -> Callable:
        """
        Returns a function for substituting a component of the Hessian of y
        for a numerical value.

        :param y_ind: the component of y whose Hessian to substitute for
        :param x_axis1: the first x axis denoting the element of the gradient
            to substitute for
        :param x_axis2: the second x axis denoting the element of the gradient
            to substitute for
        :return: the substitution function for the Hessian of y
        """

    @abstractmethod
    def y_divergence(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool) -> Callable:
        """
        Returns a function for substituting the divergence of a set of
        components of y for a numerical value.

        :param y_indices: the components of y whose divergence is to be
            substituted for
        :param indices_contiguous: whether the indices are contiguous
        :return: the substitution function for the divergence of y
        """

    @abstractmethod
    def y_curl(
            self,
            y_indices: Sequence[int],
            indices_contiguous: bool,
            curl_ind: int) -> Callable:
        """
        Returns a function for substituting the curl of a set of components of
        y for a numerical value.

        :param y_indices: the components of y whose curl is to be substituted
            for
        :param indices_contiguous: whether the indices are contiguous
        :param curl_ind: the index of the component of the curl to substitute
            for
        :return: the substitution function for the curl of y
        """

    @abstractmethod
    def y_laplacian(self, y_ind: int) -> Callable:
        """
        Returns a function for substituting a component of the Hessian of y
        for a numerical value.

        :param y_ind: the component of y whose Laplacian to substitute for
        :return: the substitution function for the Laplacian of y
        """

    def create_symbol_map(self) -> Dict[sp.Symbol, Callable]:
        """
        Creates a dictionary mapping the symbols present in the differential
        equation instance associated with the symbol mapper to a set of
        functions used to substitute the symbols for numerical values.
        """
        symbol_map = {}

        x_dimension = self._diff_eq.x_dimension
        eq_sys = self._diff_eq.symbolic_equation_system
        symbols = set()
        for lhs_type in Lhs:
            symbols.update(eq_sys.symbols_by_type(lhs_type))

        for symbol in symbols:
            symbol_name_tokens = symbol.name.split('_')
            prefix = symbol_name_tokens[0]
            indices = [int(ind) for ind in symbol_name_tokens[1:]] \
                if len(symbol_name_tokens) > 1 else []

            if prefix == 't':
                symbol_map[symbol] = self.t()
            elif prefix == 'y':
                symbol_map[symbol] = self.y(*indices)
            elif prefix == 'y-gradient':
                symbol_map[symbol] = self.y_gradient(*indices)
            elif prefix == 'y-hessian':
                symbol_map[symbol] = self.y_hessian(*indices)
            elif prefix == 'y-laplacian':
                symbol_map[symbol] = self.y_laplacian(*indices)
            else:
                indices_contiguous = np.all([
                    indices[i] == indices[i + 1] - 1
                    for i in range(len(indices) - 1)
                ])

                if prefix == 'y-divergence':
                    symbol_map[symbol] = self.y_divergence(
                        indices, indices_contiguous)
                elif prefix == 'y-curl':
                    if x_dimension == 2:
                        symbol_map[symbol] = self.y_curl(
                            indices, indices_contiguous, 0)
                    else:
                        y_indices = indices[:-1]
                        curl_ind = indices[-1]
                        symbol_map[symbol] = self.y_curl(
                            y_indices, indices_contiguous, curl_ind)

        return symbol_map
