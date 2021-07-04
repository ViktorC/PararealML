from abc import ABC
from typing import Optional, Union, Tuple

import numpy as np
from deepxde import Model as PINNModel

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator
from pararealml.core.solution import Solution
from pararealml.utils.io import suppress_stdout
from pararealml.utils.ml import RegressionModel


class MLOperator(Operator, ABC):
    """
    A base class for machine learning operators for solving differential
    equations.
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
        self._model: Optional[Union[RegressionModel, PINNModel]] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

    @property
    def model(self) -> Optional[Union[RegressionModel, PINNModel]]:
        return self._model

    @model.setter
    def model(self, model: Optional[Union[RegressionModel, PINNModel]]):
        self._model = model

    @suppress_stdout
    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True
    ) -> Solution:
        if self._model is None:
            raise ValueError

        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)

        y_shape = cp.y_shape(self._vertex_oriented)

        x = self._create_input_placeholder(cp)
        x = np.concatenate(
            (x, np.empty((x.shape[0], diff_eq.y_dimension))),
            axis=-1)
        y = np.empty((len(time_points) - 1,) + y_shape)

        y_i = ivp \
            .initial_condition \
            .discrete_y_0(self._vertex_oriented) \
            .reshape(-1, diff_eq.y_dimension)

        for i, t_i in enumerate(time_points[:-1]):
            x[:, diff_eq.x_dimension] = t_i
            x[:, diff_eq.x_dimension + 1:] = y_i
            y_i = self._model.predict(x).reshape(
                x.shape[0], diff_eq.y_dimension)
            y[i, ...] = y_i.reshape(y_shape)

        return Solution(
            cp,
            time_points[1:],
            y,
            vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)

    @staticmethod
    def model_input_shape(ivp: InitialValueProblem) -> Tuple[int]:
        """
        Returns the shape of the input of the model for the provided IVP.
        :param ivp: the initial value problem to solve
        :return: the expected input shape
        """
        diff_eq = ivp.constrained_problem.differential_equation
        return diff_eq.x_dimension + 1 + diff_eq.y_dimension,

    @staticmethod
    def model_output_shape(ivp: InitialValueProblem) -> Tuple[int]:
        """
        Returns the shape of the output of the model for the provided IVP.
        :param ivp: the initial value problem to solve
        :return: the expected output shape
        """
        diff_eq = ivp.constrained_problem.differential_equation
        return diff_eq.y_dimension,

    def _create_input_placeholder(
            self,
            cp: ConstrainedProblem
    ) -> np.ndarray:
        """
        Creates a placeholder array for the ML model inputs. If the constrained
        problem is an ODE, it returns an empty array of shape (1, 1) into which
        t can be substituted to create x. If the constrained problem is a PDE,
        it returns an array of shape (n_mesh_points, x_dimension + 1) whose
        each row is populated with the spatial coordinates of the corresponding
        mesh point in addition to an empty column for t.
        :param cp: the constrained problem to base the inputs on
        :return: the placeholder array for the ML inputs
        """
        diff_eq = cp.differential_equation

        if diff_eq.x_dimension:
            mesh = cp.mesh
            mesh_shape = mesh.shape(self._vertex_oriented)
            n_points = np.prod(mesh_shape)
            x = np.empty((n_points, diff_eq.x_dimension + 1))
            for row_ind, index in enumerate(np.ndindex(mesh_shape)):
                x[row_ind, :-1] = mesh.x(index, self._vertex_oriented)
        else:
            x = np.empty((1, 1))

        return x

    def _create_input_batch(
            self,
            cp: ConstrainedProblem,
            time_points: np.ndarray
    ) -> np.ndarray:
        """
        Creates a 2D array of inputs with a shape of
        (n_mesh_points * n_time_points, x_dimension + 1).
        :param cp: the constrained problem to base the inputs on
        :param time_points: the discretised time domain of the IVP to create
            inputs for
        :return: a batch of all inputs
        """
        input_placeholder = self._create_input_placeholder(cp)
        n_mesh_points = input_placeholder.shape[0]

        x = np.tile(input_placeholder, (len(time_points), 1))
        t = np.repeat(time_points, n_mesh_points)
        x[:, cp.differential_equation.x_dimension] = t

        return x