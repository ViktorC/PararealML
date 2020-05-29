from typing import List, Sequence, Dict, Optional

import numpy as np
from sympy import Expr

from src.core.diff_eq import DiffEq, SymbolName
from src.core.differentiator import Differentiator
from src.core.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    def discretise_time_domain(
            self,
            t_a: float,
            t_b: float) -> np.ndarray:
        """
        Returns a discretisation of the the interval [t_a, t_b^) using the
        temporal step size of the operator d_t, where t_b^ is t_b rounded to
        the nearest multiple of d_t.

        :param t_a: the beginning of the time interval
        :param t_b: the end of the time interval
        :return: the array containing the discretised temporal domain
        """
        adjusted_t_b = self.d_t() * round(t_b / self.d_t())
        return np.arange(t_a, adjusted_t_b, self.d_t())

    def calculate_d_x(
            self,
            diff_eq: DiffEq,
            y_shape: Sequence[int]) -> Optional[Sequence[float]]:
        """
        Returns a sequence of numbers denoting the step size used to discretise
        the non-temporal domain of the differential equation along each axis.

        :param diff_eq: the differential equation
        :param y_shape: the shape of the solution estimate at any point in time
        :return: the non-temporal step sizes used by the mesh
        """
        d_x = None

        if diff_eq.x_dimension():
            d_x = []
            for i, x_range in enumerate(diff_eq.x_ranges()):
                d_x_i = (x_range[1] - x_range[0]) / (y_shape[i] - 1)
                d_x.append(d_x_i)

        return d_x

    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """
        pass

    def trace(
            self,
            diff_eq: DiffEq,
            y_a: np.ndarray,
            t_a: float,
            t_b: float) -> np.ndarray:
        """
        Returns a discretised approximation of y over (t_a, t_b].

        :param diff_eq: the differential equation whose solution's trajectory
        is to be traced
        :param y_a: y(t_a), that is the value of the differential equation's
        solution at the lower bound of the interval it is to be traced over
        :param t_a: the lower bound of the interval over which the differential
        equation's solution is to be traced (exclusive)
        :param t_b: the upper bound of the interval over which the differential
        equation's solution is to be traced (inclusive)
        :return: a sequence of floating points number representing the
        discretised solution of the differential equation y over (t_a, t_b]
        """
        pass


class MethodOfLinesOperator(Operator):
    """
    A method-of-lines operator that uses conventional differential equation
    integration.
    """

    def __init__(
            self,
            integrator: Integrator,
            differentiator: Differentiator,
            d_t: float):
        """
        :param integrator: the differential equation integrator to use
        :param differentiator: the differentiator to use
        :param d_t: the temporal step size to use
        """
        self._integrator = integrator
        self._differentiator = differentiator
        self._d_t = d_t

    @staticmethod
    def _substitute_x_terms(
            diff_eq: DiffEq,
            y_shape: Sequence[int],
            exprs: List[Expr]):
        x: List[np.ndarray] = [None] * diff_eq.x_dimension()
        for i, expr in enumerate(exprs):
            if diff_eq.x_dimension() == 0:
                x_symbol = SymbolName.x.format('')
                if expr.has(x_symbol):
                    if x[0] is None:
                        x_range = diff_eq.x_ranges()[0]
                        x[0] = np.linspace(
                            x_range[0], x_range[1], y_shape[0])
                    exprs[i] = expr.subs(x_symbol, x[0])

            for j in range(diff_eq.x_dimension()):
                x_j_symbol = SymbolName.x.format(i)
                if expr.has(x_j_symbol):
                    if x[j] is None:
                        x_range = diff_eq.x_ranges()[j]
                        x[j] = np.linspace(
                            x_range[0], x_range[1], y_shape[j])
                    exprs[i] = expr.subs(x_j_symbol, x[j])

    @staticmethod
    def _substitute_t_term(
            expr: Expr,
            t: float) -> Expr:
        if expr.has(SymbolName.t):
            expr = expr.subs(SymbolName.t, t)

        return expr

    @staticmethod
    def _substitute_y_term(
            expr: Expr,
            y: np.ndarray,
            y_ind: int,
            y_suffix: str) -> Expr:
        y_symbol = SymbolName.y.format(y_suffix)
        if expr.has(y_symbol):
            expr = expr.subs(y_symbol, y[..., y_ind])

        return expr

    def _substitute_d_y_wrt_x(
            self,
            expr: Expr,
            y: np.ndarray,
            d_x: float,
            x_ind: int,
            x_suffix: str,
            y_ind: int,
            y_suffix: str,
            cache: Dict[str, np.ndarray]) -> Expr:
        d_y_wrt_x_symbol = SymbolName.d_y_wrt_x.format(x_suffix, y_suffix)
        if expr.has(d_y_wrt_x_symbol):
            if d_y_wrt_x_symbol not in cache:
                d_y_wrt_x = self._differentiator.derivative(
                    y, d_x, x_ind, y_ind)
                cache[d_y_wrt_x_symbol] = d_y_wrt_x
            else:
                d_y_wrt_x = cache[d_y_wrt_x_symbol]

            expr = expr.subs(d_y_wrt_x_symbol, d_y_wrt_x)

        return expr

    def _substitute_d2_y_wrt_x(
            self,
            expr: Expr,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_ind1: int,
            x_ind2: int,
            x_suffix1: str,
            x_suffix2: str,
            y_ind: int,
            y_suffix: str,
            cache: Dict[str, np.ndarray]) -> Expr:
        d2_y_wrt_x_symbol = SymbolName.d2_y_wrt_x.format(
            x_suffix1, x_suffix2, y_suffix)
        if expr.has(d2_y_wrt_x_symbol):
            if d2_y_wrt_x_symbol not in cache:
                d2_y_wrt_x = self._differentiator.second_derivative(
                    y, d_x1, d_x2, x_ind1, x_ind2, y_ind)
                cache[d2_y_wrt_x_symbol] = d2_y_wrt_x
            else:
                d2_y_wrt_x = cache[d2_y_wrt_x_symbol]

            expr = expr.subs(d2_y_wrt_x_symbol, d2_y_wrt_x)

        return expr

    def _substitute_y_x_terms(
            self,
            expr: Expr,
            y: np.ndarray,
            d_x: Sequence[float],
            y_ind: int,
            y_suffix: str,
            cache: Dict[str, np.ndarray]) -> Expr:
        if len(y.shape) == 2:
            expr = self._substitute_d_y_wrt_x(
                expr, y, d_x[0], 0, '', y_ind, y_suffix, cache)
            expr = self._substitute_d2_y_wrt_x(
                expr, y, d_x[0], d_x[0], 0, 0, '', '', y_ind, y_suffix, cache)

        for i in range(len(y.shape) - 1):
            expr = self._substitute_d_y_wrt_x(
                expr, y, d_x[i], i, str(i), y_ind, y_suffix, cache)

            for j in range(len(y.shape) - 1):
                expr = self._substitute_d2_y_wrt_x(
                    expr, y, d_x[i], d_x[j], i, j, str(i), str(j),
                    y_ind, y_suffix, cache)

        grad_y_symbol = SymbolName.grad_y.format(y_suffix)
        if expr.has(grad_y_symbol):
            if grad_y_symbol not in cache:
                grad_y = self._differentiator.gradient(y[..., y_ind], d_x)
                cache[grad_y_symbol] = grad_y
            else:
                grad_y = cache[grad_y_symbol]

            expr = expr.subs(grad_y_symbol, grad_y)

        del2_y_symbol = SymbolName.del2_y.format(y_suffix)
        if expr.has(del2_y_symbol):
            if del2_y_symbol not in cache:
                del2_y = self._differentiator.laplacian(y[..., y_ind], d_x)
                cache[del2_y_symbol] = del2_y
            else:
                del2_y = cache[del2_y_symbol]

            expr = expr.subs(del2_y_symbol, del2_y)

        return expr

    def _substitute_dynamic_terms_and_eval_exprs_ode(
            self,
            exprs: List[Expr],
            y: np.ndarray,
            t: float) -> np.ndarray:
        eval = np.empty(len(exprs))

        for i, expr in enumerate(exprs):
            expr = self._substitute_t_term(expr, t)

            if len(exprs) == 1:
                expr = self._substitute_y_term(expr, y, 0, '')

            for j in range(len(exprs)):
                expr = self._substitute_y_term(expr, y, j, str(j))

            assert len(expr.free_symbols) == 0
            eval[i] = expr

        return eval

    def _substitute_dynamic_terms_and_eval_exprs_pde(
            self,
            exprs: List[Expr],
            y: np.ndarray,
            t: float,
            d_x: Sequence[float]) -> np.ndarray:
        eval = np.empty(len(exprs))
        cache = {}

        for i, expr in enumerate(exprs):
            expr = self._substitute_t_term(expr, t)

            if expr.has(SymbolName.div_y):
                if SymbolName.div_y not in cache:
                    div_y = self._differentiator.divergence(y, d_x)
                    cache[SymbolName.div_y] = div_y
                else:
                    div_y = cache[SymbolName.div_y]

                expr = expr.subs(SymbolName.div_y, div_y)

            if expr.has(SymbolName.curl_y):
                if SymbolName.curl_y not in cache:
                    curl_y = self._differentiator.curl(y, d_x)
                    cache[SymbolName.curl_y] = curl_y
                else:
                    curl_y = cache[SymbolName.curl_y]

                expr = expr.subs(SymbolName.curl_y, curl_y)

            if len(exprs) == 1:
                expr = self._substitute_y_term(expr, y, 0, '')
                expr = self._substitute_y_x_terms(expr, y, d_x, 0, '', cache)

            for j in range(len(exprs)):
                expr = self._substitute_y_term(expr, y, j, str(j))
                expr = self._substitute_y_x_terms(
                    expr, y, d_x, j, str(j), cache)

            assert len(expr.free_symbols) == 0
            eval[i] = expr

        return eval

    def _create_d_y_wrt_t(self, diff_eq: DiffEq, y_shape: Sequence[int]):
        exprs = list(diff_eq.d_y())

        if diff_eq.x_dimension():
            d_x = self.calculate_d_x(diff_eq, y_shape)
            self._substitute_x_terms(diff_eq, y_shape, exprs)

            def d_y_wrt_t(t: float, y: np.ndarray) -> np.ndarray:
                return self._substitute_dynamic_terms_and_eval_exprs_pde(
                    exprs, y, t, d_x)
        else:
            def d_y_wrt_t(t: float, y: np.ndarray) -> np.ndarray:
                return self._substitute_dynamic_terms_and_eval_exprs_ode(
                    exprs, y, t)

        return d_y_wrt_t

    def d_t(self) -> float:
        return self._d_t

    def trace(
            self,
            diff_eq: DiffEq,
            y_a: np.ndarray,
            t_a: float,
            t_b: float) -> np.ndarray:
        t = self.discretise_time_domain(t_a, t_b)

        d_y_wrt = self._create_d_y_wrt_t(diff_eq, y_a.shape)

        y_shape = list(y_a.shape)
        y_shape.insert(0, len(t))
        y = np.empty(tuple(y_shape))

        y_i = y_a

        for i, t_i in enumerate(t):
            y_i = self._integrator.integral(y_i, t_i, self._d_t, d_y_wrt)
            y[i] = y_i

        return y


# class MLOperator(Operator):
#     """
#     A machine learning accelerated operator that uses a regression model to
#     integrate differential equations.
#     """
#
#     def __init__(
#             self, model: Any, d_t: float):
#         """
#         :param model: the regression model to use as the integrator; its input
#         are the values of t, y(t), and y'(t) and its output is y(t + d_t) where
#         d_t is the step size of this operator defined by the corresponding
#         constructor argument
#         :param d_t: the step size of the operator; it determines the lengths of
#         the domain slices over which the training operator is used to trace the
#         differential equation's solution and provide the labels for the
#         training data
#         """
#         self._model = model
#         self._d_t = d_t
#         self._trained: bool = False
#
#     def train_model(
#             self,
#             diff_eq: DiffEq,
#             trainer: Operator,
#             data_epochs: int,
#             y_noise_var_coeff: float = 1.):
#         """
#         Trains the regression model behind the operator on the provided
#         differential equation.
#
#         It generates the training data by repeatedly iterating over the domain
#         of the differential equation in steps of size d_t and tracing the
#         solution using the training operator. At every step i, a new training
#         data point is created out of the values of t_i, y(t_i), and y'(t_i)
#         labelled by y(t_i+1) = y(t_i + d_t) as estimated by the training
#         operator. Once the data point is created, a 0-mean Gaussian noise is
#         added to the value of y(t_i+1) to perturbate the trajectory of y.
#         This introduces some variance to the training data and helps better
#         approximate the function represented by the training operator. The
#         standard deviation of this Gaussian is c^(1/2) * y'(t_i) * d_t where
#         c is the noise variance coefficient.
#
#         :param diff_eq: the differential equation to train the model on
#         :param trainer: the operator for generating the labels for the training
#         data
#         :param data_epochs: the number of iterations to perform over the domain
#         of the differential equation to generate the training data
#         :param y_noise_var_coeff: the noise variance coefficient that
#         determines the amount of perturbation to apply to the trajectory of
#         the trainer operator's solution
#         """
#         t = self._discretise_time_domain(diff_eq.t_min(), diff_eq.t_max())
#         if diff_eq.y_dimension() == 1:
#             obs = np.empty((data_epochs * len(t), 3))
#             y = np.empty(len(obs))
#         else:
#             obs = np.empty((
#                 data_epochs * len(t),
#                 1 + 2 * diff_eq.y_dimension()))
#             y = np.empty((len(obs), diff_eq.y_dimension()))
#
#         for k in range(data_epochs):
#             offset = k * len(t)
#             y_i = diff_eq.y_0()
#             for i, t_i in enumerate(t):
#                 ind = offset + i
#                 y[ind] = trainer.trace(
#                     diff_eq, y_i, t_i, t_i + self._d_t)[-1]
#                 d_y_i = diff_eq.d_y(t_i, y_i)
#                 obs[ind][0] = t_i
#
#                 if diff_eq.y_dimension() == 1:
#                     obs[ind][1] = y_i
#                     obs[ind][2] = d_y_i
#                     y_i = y[ind] + np.random.normal(
#                         0.,
#                         math.sqrt(y_noise_var_coeff) * d_y_i * self._d_t)
#                 else:
#                     obs[ind][1:1 + diff_eq.y_dimension()] = y_i
#                     obs[ind][1 + diff_eq.y_dimension():] = d_y_i
#                     y_i = y[ind] + np.random.multivariate_normal(
#                         np.zeros(diff_eq.y_dimension()),
#                         np.diag(d_y_i * (y_noise_var_coeff * self._d_t)))
#
#         self._model.fit(obs, y)
#         self._trained = True
#
#     def d_t(self) -> float:
#         return self._d_t
#
#     def trace(
#             self,
#             diff_eq: DiffEq,
#             y_a: ImageType,
#             t_a: float,
#             t_b: float) -> ImageType:
#         assert self._trained
#
#         t = self._discretise_time_domain(t_a, t_b)
#         if diff_eq.y_dimension() == 1:
#             x = np.empty((1, 3))
#             y = np.empty(len(t))
#         else:
#             x = np.empty((1, 1 + 2 * diff_eq.y_dimension()))
#             y = np.empty((len(t), diff_eq.y_dimension()))
#
#         y_i = y_a
#
#         for i, t_i in enumerate(t):
#             x[0, 0] = t_i
#             x[0, 1:1 + diff_eq.y_dimension()] = y_i
#             x[0, 1 + diff_eq.y_dimension():] = diff_eq.d_y(t_i, y_i)
#             y_i = self._model.predict(x)[0]
#             y[i] = y_i
#
#         return y
