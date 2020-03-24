from typing import Sequence

import numpy as np

from src.diff_eq import OrdinaryDiffEq
from src.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential equation over a specific domain interval
    given an initial value.
    """

    """
    Returns the step size of the operator.
    """
    def d_x(self) -> float:
        pass

    """
    Returns a discretised approximation of y over (x_a, x_b].
    """
    def trace(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> Sequence[float]:
        pass


class ConventionalOperator(Operator):
    """
    An operator that uses conventional differential equation integration.
    """

    def __init__(self, integrator: Integrator, d_x: float):
        self._integrator = integrator
        self._d_x = d_x

    def d_x(self) -> float:
        return self._d_x

    def trace(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> Sequence[float]:
        x = np.arange(x_a, x_b, self._d_x)
        y = np.empty(len(x))
        y_i = y_a

        for i, x_i in enumerate(x):
            y_i = self._integrator.integrate(y_i, x_i, self._d_x, diff_eq.d_y)
            y[i] = y_i

        return y


class MLOperator(Operator):
    """
    A machine learning accelerated operator that uses a regression model to integrate differential equations.
    """

    def __init__(self, model, trainer: ConventionalOperator, d_x: float):
        self._model = model
        self._trainer = trainer
        self._d_x = d_x
        self._diff_eq_trained_on = None

    """
    Trains the regression model behind the operator on the provided differential equation.
    """
    def train_model(self, diff_eq: OrdinaryDiffEq):
        x = np.arange(diff_eq.x_0(), diff_eq.x_max() + self._d_x / 2, self._d_x)
        obs = np.empty((len(x) - 1, 2))
        y = np.empty(len(obs))
        y_i = diff_eq.y_0()

        for i, x_i in enumerate(x[:-1]):
            obs[i][0] = y_i
            obs[i][1] = diff_eq.d_y(x_i, y_i)
            y_i = self._trainer.trace(diff_eq, y_i, x_i, x[i + 1])[-1]
            y[i] = y_i

        self._model.fit(obs, y)
        self._diff_eq_trained_on = diff_eq

    def d_x(self) -> float:
        return self._d_x

    def trace(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> Sequence[float]:
        if diff_eq != self._diff_eq_trained_on:
            self.train_model(diff_eq)

        x = np.arange(x_a, x_b, self._d_x)
        y = np.empty(len(x))
        y_i = y_a

        for i, x_i in enumerate(x):
            y_i = self._model.predict([[y_i, diff_eq.d_y(x_i, y_i)]])[0]
            y[i] = y_i

        return y
