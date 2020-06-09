from typing import Callable

import numpy as np

from src.core.differentiator import Differentiator
from src.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from src.core.integrator import Integrator

SolutionConstraintFunction = Callable[[np.ndarray], None]


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    def _discretise_time_domain(self, t: TemporalDomainInterval) -> np.ndarray:
        """
        Returns a discretisation of the the interval [t_a, t_b^) using the
        temporal step size of the operator d_t, where t_b^ is t_b rounded to
        the nearest multiple of d_t.

        :param t: the time interval to discretise
        :return: the array containing the discretised temporal domain
        """
        adjusted_t_1 = self.d_t() * round(t[1] / self.d_t())
        return np.arange(t[0], adjusted_t_1, self.d_t())

    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """
        pass

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        """
        Returns a discretised approximation of the IVP's solution.

        :param ivp: the initial value problem to solve
        :return: the discretised solution of the IVP
        """
        pass


class FDMOperator(Operator):
    """
    A finite difference method based conventional differential equation solver.
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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem()
        diff_eq = bvp.differential_equation()
        d_x = bvp.mesh().d_x()
        y_constraint_function = bvp.y_constraint_function()
        d_y_constraint_function = bvp.d_y_constraint_function()

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return diff_eq.d_y_over_d_t(
                _t,
                _y,
                d_x,
                self._differentiator,
                d_y_constraint_function)

        time_steps = self._discretise_time_domain(ivp.t_interval())

        y = np.empty([len(time_steps)] + list(bvp.y_shape()))
        y_i = ivp.y_0()

        for i, t_i in enumerate(time_steps):
            y_i = self._integrator.integral(y_i, t_i, self._d_t, d_y_over_d_t)
            y_constraint_function(y_i)
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
