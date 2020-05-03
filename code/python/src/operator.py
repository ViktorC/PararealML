from typing import Any, Optional

import numpy as np

from src.diff_eq import DiffEq, ImageType
from src.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    def _discretise_time_domain(
            self,
            t_a: float,
            t_b: float) -> np.ndarray:
        """
        Returns a discretisation of the the interval [t_a, t_b^) using the
        operator's step size, d_t, where t_b^ is t_b rounded to the nearest
        multiple of d_t.
        """
        adjusted_t_b = self.d_t() * round(t_b / self.d_t())
        return np.arange(t_a, adjusted_t_b, self.d_t())

    def d_t(self) -> float:
        """
        Returns the step size of the operator.
        """
        pass

    def trace(
            self,
            diff_eq: DiffEq,
            y_a: ImageType,
            t_a: float,
            t_b: float) -> ImageType:
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


class ConventionalOperator(Operator):
    """
    An operator that uses conventional differential equation integration.
    """

    def __init__(self, integrator: Integrator, d_t: float):
        """
        :param integrator: the differential equation integrator to use
        :param d_t: the step size to use with the integrator
        """
        self._integrator = integrator
        self._d_t = d_t

    def d_t(self) -> float:
        return self._d_t

    def trace(
            self,
            diff_eq: DiffEq,
            y_a: ImageType,
            t_a: float,
            t_b: float) -> ImageType:
        t = self._discretise_time_domain(t_a, t_b)
        if isinstance(y_a, float):
            y = np.empty(len(t))
        else:
            y = np.empty((len(t), len(y_a)))
        y_i = y_a

        for i, t_i in enumerate(t):
            y_i = self._integrator.integrate(y_i, t_i, self._d_t, diff_eq.d_y)
            y[i] = y_i

        return y


class MLOperator(Operator):
    """
    A machine learning accelerated operator that uses a regression model to
    integrate differential equations.
    """

    def __init__(
            self, model: Any, trainer: Operator, d_t: float, data_epochs: int):
        """
        :param model: the regression model to use as the integrator; its input
        are the values of t, y(t), and y'(t) and its output is y(t + d_t) where
        d_t is the step size of this operator defined by the corresponding
        constructor argument
        :param trainer: the operator for generating the labels for the data
        to train the regression model on
        :param d_t: the step size of the operator; it determines the lengths of
        the domain slices over which the training operator is used to trace the
        differential equation's solution and provide the labels for the
        training data
        :param data_epochs: the number of iterations to perform over the domain
        of the differential equation to generate the training data
        """
        self._model = model
        self._trainer = trainer
        self._d_t = d_t
        self._data_epochs = data_epochs
        self._trained: bool = False

    def train_model(self, diff_eq: DiffEq):
        """
        Trains the regression model behind the operator on the provided
        differential equation.

        It generates the training data by repeatedly iterating over the domain
        of the differential equation in steps of size d_t and tracing the
        solution using the training operator. At every step i, a new training
        data point is created out of the values of t_i, y(t_i), and y'(t_i)
        labelled by y(t_i+1) = y(t_i + d_t) as estimated by the training
        operator. Once the data point is created, a 0-mean Gaussian noise is
        added to the value of y(t_i+1) to perturbate the trajectory of y.
        This introduces some variance to the training data and helps better
        approximate the function represented by the training operator. The
        standard deviation of this Gaussian is y'(t_i) * d_t.

        :param diff_eq: the differential equation to train the model on
        """
        t = self._discretise_time_domain(diff_eq.t_0(), diff_eq.t_max())

        y_0 = diff_eq.y_0()
        if isinstance(y_0, float):
            obs = np.empty((self._data_epochs * len(t), 3))
            y = np.empty(len(obs))
        else:
            obs = np.empty((self._data_epochs * len(t), 1 + 2 * len(y_0)))
            y = np.empty((len(obs), len(y_0)))

        for k in range(self._data_epochs):
            offset = k * len(t)
            y_i = y_0
            for i, t_i in enumerate(t):
                ind = offset + i
                y[ind] = self._trainer.trace(
                    diff_eq, y_i, t_i, t_i + self._d_t)[-1]
                d_y_i = diff_eq.d_y(t_i, y_i)
                obs[ind][0] = t_i

                if isinstance(y_0, float):
                    obs[ind][1] = y_i
                    obs[ind][2] = d_y_i
                    y_i = y[ind] + np.random.normal(0., d_y_i * self._d_t)
                else:
                    obs[ind][1:1 + len(y_0)] = y_i
                    obs[ind][1 + len(y_0):] = d_y_i
                    y_i = y[ind] + np.random.multivariate_normal(
                        np.zeros(len(y_0)), np.diag(d_y_i * self._d_t))

        self._model.fit(obs, y)
        self._trained = True

    def d_t(self) -> float:
        return self._d_t

    def trace(
            self,
            diff_eq: DiffEq,
            y_a: ImageType,
            t_a: float,
            t_b: float) -> ImageType:
        assert self._trained

        t = self._discretise_time_domain(t_a, t_b)
        if isinstance(y_a, float):
            x = np.empty((1, 3))
            y = np.empty(len(t))
        else:
            x = np.empty((1, 1 + 2 * len(y_a)))
            y = np.empty((len(t), len(y_a)))
        y_i = y_a

        for i, t_i in enumerate(t):
            x[0, 0] = t_i
            x[0, 1:1 + len(y_a)] = y_i
            x[0, 1 + len(y_a):] = diff_eq.d_y(t_i, y_i)
            y_i = self._model.predict(x)[0]
            y[i] = y_i

        return y
