from typing import Sequence, Any

import numpy as np

from src.diff_eq import OrdinaryDiffEq
from src.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific domain interval given an initial value.
    """

    def d_x(self) -> float:
        """
        Returns the step size of the operator.
        """
        pass

    def trace(
            self,
            diff_eq: OrdinaryDiffEq,
            y_a: float,
            x_a: float,
            x_b: float) -> Sequence[float]:
        """
        Returns a discretised approximation of y over (x_a, x_b].

        :param diff_eq: the differential equation whose solution's trajectory
        is to be traced
        :param y_a: y(x_a), that is the value of the differential equation's
        solution at the lower bound of the interval it is to be traced over
        :param x_a: the lower bound of the interval over which the differential
        equation's solution is to be traced (exclusive)
        :param x_b: the upper bound of the interval over which the differential
        equation's solution is to be traced (inclusive)
        :return: a sequence of floating points number representing the
        discretised solution of the differential equation y over (x_a, x_b]
        """
        pass


class ConventionalOperator(Operator):
    """
    An operator that uses conventional differential equation integration.
    """

    def __init__(self, integrator: Integrator, d_x: float):
        """
        :param integrator: the differential equation integrator to use
        :param d_x: the step size to use with the integrator
        """
        self._integrator = integrator
        self._d_x = d_x

    def d_x(self) -> float:
        return self._d_x

    def trace(
            self,
            diff_eq: OrdinaryDiffEq,
            y_a: float,
            x_a: float,
            x_b: float) -> Sequence[float]:
        x = np.arange(x_a, x_b, self._d_x)
        y = np.empty(len(x))
        y_i = y_a

        for i, x_i in enumerate(x):
            y_i = self._integrator.integrate(y_i, x_i, self._d_x, diff_eq.d_y)
            y[i] = y_i

        return y


class MLOperator(Operator):
    """
    A machine learning accelerated operator that uses a regression model to
    integrate differential equations.
    """

    def __init__(
            self, model: Any, trainer: Operator, d_x: float, data_epochs: int):
        """
        :param model: the regression model to use as the integrator; its input
        are the values of x, y(x), and y'(x) and its output is y(x + d_x) where
        d_x is the step size of this operator defined by the corresponding
        constructor argument
        :param trainer: the operator for generating the labels for the data
        to train the regression model on
        :param d_x: the step size of the operator; it determines the lengths of
        the domain slices over which the training operator is used to trace the
        differential equation's solution and provide the labels for the
        training data
        :param data_epochs: the number of iterations to perform over the domain
        of the differential equation to generate the training data
        """
        self._model = model
        self._trainer = trainer
        self._d_x = d_x
        self._data_epochs = data_epochs
        self._diff_eq_trained_on = None

    def train_model(self, diff_eq: OrdinaryDiffEq):
        """
        Trains the regression model behind the operator on the provided
        differential equation.

        It generates the training data by repeatedly iterating over the domain
        of the differential equation in steps of size d_x and tracing the
        solution using the training operator. At every step i, a new training
        data point is created out of the values of x_i, y(x_i), and y'(x_i)
        labelled by y(x_i+1) = y(x_i + d_x) as estimated by the training
        operator. Once the data point is created, a 0-mean Gaussian noise is
        added to the value of y(x_i+1) to perturbate the trajectory of y.
        This introduces some variance to the training data and helps better
        approximate the function represented by the training operator. The
        standard deviation of this Gaussian is y'(x_i) * d_x.

        :param diff_eq: the differential equation to train the model on
        """
        x = np.arange(
            diff_eq.x_0(), diff_eq.x_max() + self._d_x / 2, self._d_x)
        obs = np.empty((self._data_epochs * (len(x) - 1), 3))
        y = np.empty(len(obs))

        for k in range(self._data_epochs):
            offset = k * (len(x) - 1)
            y_i = diff_eq.y_0()
            for i, x_i in enumerate(x[:-1]):
                ind = offset + i
                d_y_i = diff_eq.d_y(x_i, y_i)
                obs[ind][0] = x_i
                obs[ind][1] = y_i
                obs[ind][2] = d_y_i
                y[ind] = self._trainer.trace(diff_eq, y_i, x_i, x[i + 1])[-1]
                y_i = y[ind] + np.random.normal(0., d_y_i * self._d_x)

        self._model.fit(obs, y)
        self._diff_eq_trained_on = diff_eq

    def d_x(self) -> float:
        return self._d_x

    def trace(
            self,
            diff_eq: OrdinaryDiffEq,
            y_a: float,
            x_a: float,
            x_b: float) -> Sequence[float]:
        if diff_eq != self._diff_eq_trained_on:
            self.train_model(diff_eq)

        x = np.arange(x_a, x_b, self._d_x)
        y = np.empty(len(x))
        y_i = y_a

        for i, x_i in enumerate(x):
            y_i = self._model.predict([[x_i, y_i, diff_eq.d_y(x_i, y_i)]])[0]
            y[i] = y_i

        return y
