from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)

import numpy as np
import sympy as sp

from pararealml.differential_equation import LHS, DifferentialEquation

SymbolMapArg = TypeVar("SymbolMapArg")
SymbolMapValue = TypeVar("SymbolMapValue")
SymbolMapFunction = Callable[[SymbolMapArg], SymbolMapValue]


class SymbolMapper(ABC, Generic[SymbolMapArg, SymbolMapValue]):
    """
    A class for mapping symbolic differential equation to numerical values.
    """

    def __init__(self, diff_eq: DifferentialEquation):
        """
        :param diff_eq: the differential equation to create a symbol mapper for
        """
        self._diff_eq = diff_eq
        self._symbol_map = self.create_symbol_map()

        eq_sys = diff_eq.symbolic_equation_system
        self._rhs_functions: Dict[
            Optional[LHS], Callable[[SymbolMapArg], Sequence[SymbolMapValue]]
        ] = {None: self.create_rhs_map_function(range(len(eq_sys.rhs)))}
        for lhs_type in LHS:
            self._rhs_functions[lhs_type] = self.create_rhs_map_function(
                eq_sys.equation_indices_by_type(lhs_type)
            )

    @abstractmethod
    def t_map_function(self) -> SymbolMapFunction:
        """
        Returns a function for mapping t to a numerical value.
        """

    @abstractmethod
    def y_map_function(self, y_ind: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of y to a numerical value.

        :param y_ind: the component of y to return a map for
        :return: the mapper function for y
        """

    @abstractmethod
    def x_map_function(self, x_axis: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of x to a numerical value.

        :param x_axis: the component of x to return a map for
        :return: the mapper function for x
        """

    @abstractmethod
    def y_gradient_map_function(
        self, y_ind: int, x_axis: int
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of the gradient of y to a
        numerical value.

        :param y_ind: the component of y whose gradient to return a map for
        :param x_axis: the x-axis denoting the element of the gradient to
            return a map for
        :return: the mapper function for the gradient of y
        """

    @abstractmethod
    def y_hessian_map_function(
        self, y_ind: int, x_axis1: int, x_axis2: int
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of the Hessian of y to a
        numerical value.

        :param y_ind: the component of y whose Hessian to return a map for
        :param x_axis1: the first x-axis denoting the element of the gradient
            to return a map for
        :param x_axis2: the second x-axis denoting the element of the gradient
            to return a map for
        :return: the mapper function for the Hessian of y
        """

    @abstractmethod
    def y_divergence_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping the divergence of a set of components of
        y to a numerical value.

        :param y_indices: the components of y whose divergence to return a map
            for
        :param indices_contiguous: whether the indices are contiguous
        :return: the mapper function for the divergence of y
        """

    @abstractmethod
    def y_curl_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        curl_ind: int,
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping the curl of a set of components of y to
        a numerical value.

        :param y_indices: the components of y whose curl to return a map for
        :param indices_contiguous: whether the indices are contiguous
        :param curl_ind: the index of the component of the curl to map
        :return: the mapper function for the curl of y
        """

    @abstractmethod
    def y_laplacian_map_function(self, y_ind: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of the element-wise scalar
        Laplacian of y to a numerical value.

        :param y_ind: the component of y whose Laplacian to return a mp for
        :return: the mapper function for the Laplacian of y
        """

    @abstractmethod
    def y_vector_laplacian_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        vector_laplacian_ind: int,
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping the vector Laplacian of a set of
        components of y to a numerical value.

        :param y_indices: the components of y whose vector Laplacian to return
            a map for
        :param indices_contiguous: whether the indices are contiguous
        :param vector_laplacian_ind: the index of the component of the vector
            Laplacian to map
        :return: the mapper function for the vector Laplacian of y
        """

    def create_symbol_map(self) -> Dict[sp.Basic, SymbolMapFunction]:
        """
        Creates a dictionary linking the symbols present in the differential
        equation instance associated with the symbol mapper to a set of
        functions used to map the symbols to numerical values.
        """
        symbol_map: Dict[sp.Basic, Callable] = {}

        x_dimension = self._diff_eq.x_dimension
        eq_sys = self._diff_eq.symbolic_equation_system
        all_symbols = set.union(*[rhs.free_symbols for rhs in eq_sys.rhs])

        for symbol in all_symbols:
            symbol_name_tokens = symbol.name.split("_")
            prefix = symbol_name_tokens[0]
            indices = (
                [int(ind) for ind in symbol_name_tokens[1:]]
                if len(symbol_name_tokens) > 1
                else []
            )

            if prefix == "t":
                symbol_map[symbol] = self.t_map_function()
            elif prefix == "y":
                symbol_map[symbol] = self.y_map_function(*indices)
            elif prefix == "x":
                symbol_map[symbol] = self.x_map_function(*indices)
            elif prefix == "y-gradient":
                symbol_map[symbol] = self.y_gradient_map_function(*indices)
            elif prefix == "y-hessian":
                symbol_map[symbol] = self.y_hessian_map_function(*indices)
            elif prefix == "y-laplacian":
                symbol_map[symbol] = self.y_laplacian_map_function(*indices)
            else:
                indices_contiguous = np.all(
                    [
                        indices[i] == indices[i + 1] - 1
                        for i in range(len(indices) - 1)
                    ]
                )

                if prefix == "y-divergence":
                    symbol_map[symbol] = self.y_divergence_map_function(
                        indices, indices_contiguous
                    )
                elif prefix == "y-curl":
                    symbol_map[symbol] = (
                        self.y_curl_map_function(
                            indices, indices_contiguous, 0
                        )
                        if x_dimension == 2
                        else self.y_curl_map_function(
                            indices[:-1], indices_contiguous, indices[-1]
                        )
                    )
                elif prefix == "y-vector-laplacian":
                    self.y_vector_laplacian_map_function(
                        indices[:-1], indices_contiguous, indices[-1]
                    )

        return symbol_map

    def create_rhs_map_function(
        self, indices: Sequence[int]
    ) -> Callable[[SymbolMapArg], Sequence[SymbolMapValue]]:
        """
        Creates a function for evaluating the right-hand sides of the equations
        denoted by the provided indices.

        :param indices: the indices of the equations within the differential
            equation system whose evaluation function is to be created
        :return: a function that returns the numerical value of the right-hand
            sides given a substitution argument
        """
        rhs = self._diff_eq.symbolic_equation_system.rhs

        selected_rhs = []
        selected_rhs_symbols: Set[sp.Basic] = set()
        for i in indices:
            rhs_i = rhs[i]
            selected_rhs.append(rhs_i)
            selected_rhs_symbols.update(rhs_i.free_symbols)

        subst_functions = [
            self._symbol_map[symbol] for symbol in selected_rhs_symbols
        ]
        rhs_lambda = sp.lambdify([selected_rhs_symbols], selected_rhs, "numpy")

        def rhs_map_function(arg: SymbolMapArg) -> Sequence[SymbolMapValue]:
            return rhs_lambda(
                [subst_function(arg) for subst_function in subst_functions]
            )

        return rhs_map_function

    def map(
        self, arg: SymbolMapArg, lhs_type: Optional[LHS] = None
    ) -> Sequence[SymbolMapValue]:
        """
        Evaluates the right-hand side of the differential equation system
        given the map argument.

        :param arg: the map argument that the numerical values of the
            right-hand sides depend on
        :param lhs_type: the left-hand type of the equations whose right-hand
            sides are to be evaluated; if None, the whole differential equation
            system's right-hand side is evaluated
        :return: the numerical value of the right-hand side of the differential
            equation as a sequence of map values where each element corresponds
            to an equation within the system
        """
        return self._rhs_functions[lhs_type](arg)
