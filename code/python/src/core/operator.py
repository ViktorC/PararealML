from typing import Sequence, Optional

import numpy as np

from src.core.differential_equation import DifferentialEquation
from src.core.differentiator import Differentiator
from src.core.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    @staticmethod
    def _calculate_d_x(
            diff_eq: DifferentialEquation,
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

    def _discretise_time_domain(
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

    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """
        pass

    def trace(
            self,
            diff_eq: DifferentialEquation,
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

    def d_t(self) -> float:
        return self._d_t

    def trace(
            self,
            diff_eq: DifferentialEquation,
            y_a: np.ndarray,
            t_a: float,
            t_b: float) -> np.ndarray:
        t = self._discretise_time_domain(t_a, t_b)

        y_shape = list(y_a.shape)
        y_shape.insert(0, len(t))
        y = np.empty(tuple(y_shape))

        y_i = y_a

        if diff_eq.x_dimension():
            d_x = self._calculate_d_x(diff_eq, y_shape)
        else:
            d_x = None

        def d_y_wrt_t(_t: float, _y: np.ndarray, _d_x: Sequence[float]=d_x) \
                -> np.ndarray:
            return diff_eq.d_y(_t, _y, _d_x, self._differentiator)

        for i, t_i in enumerate(t):
            y_i = self._integrator.integral(y_i, t_i, self._d_t, d_y_wrt_t)
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
