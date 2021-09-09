from typing import Union, Tuple, Callable, Optional, Protocol

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_condition import DiscreteInitialCondition
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator, discretize_time_domain
from pararealml.core.solution import Solution


class SKLearnRegressor(Protocol):
    def fit(self, x, y, sample_weight=None): ...
    def predict(self, x): ...
    def score(self, x, y, sample_weight=None): ...


RegressionModel = Union[SKLearnRegressor, KerasRegressor]


class AutoRegressionOperator(Operator):
    """
    A supervised machine learning operator that uses auto regression to model
    a high fidelity operator for solving initial value problems.
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
            raise ValueError(f'time step size ({d_t}) must be greater than 0')

        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._model: Optional[RegressionModel] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

    @property
    def model(self) -> Optional[RegressionModel]:
        """
        The regression model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[RegressionModel]):
        self._model = model

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True) -> Solution:
        if self._model is None:
            raise ValueError('operator has no model')

        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        y_shape = cp.y_shape(self._vertex_oriented)

        features = self._create_input_placeholder(cp)
        t = discretize_time_domain(ivp.t_interval, self._d_t)
        y = np.empty((len(t) - 1,) + y_shape)

        y_i = ivp \
            .initial_condition \
            .discrete_y_0(self._vertex_oriented) \
            .reshape((-1, diff_eq.y_dimension))

        for i, t_i in enumerate(t[:-1]):
            features[:, diff_eq.x_dimension] = t_i
            features[:, 1 + diff_eq.x_dimension:] = y_i.reshape((1, -1))
            y_i = self._model.predict(features)
            y[i, ...] = y_i.reshape(y_shape)

        return Solution(
            ivp,
            t[1:],
            y,
            vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)

    def train(
            self,
            ivp: InitialValueProblem,
            oracle: Operator,
            model: RegressionModel,
            iterations: int,
            perturbation_function: Callable[[float, np.ndarray], np.ndarray],
            test_size: float = .2,
            score_func: Callable[[np.ndarray, np.ndarray], float] =
            mean_squared_error) -> Tuple[float, float]:
        """
        Fits a regression model to training data generated by the oracle.

        The inputs of the model are time t, spatial coordinates x, and the
        values of the solution at all points of the IVP's mesh at time t
        (for ODEs, no spatial coordinates are included in the features and the
        solution is evaluated merely at t). The model outputs are the predicted
        values of the solution at x and t + d_t (for ODEs, it is again just the
        solution at t + d_t).

        The training data is generated by using the oracle to repeatedly solve
        sub-IVPs with perturbed initial conditions and a time domain extent
        matching the step size of this operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        :param iterations: the number of data generation iterations
        :param perturbation_function: a function that takes a time argument,
            representing the start of a sub-IVP's time domain, and the discrete
            initial values for the sub-IVP's solution and returns a perturbed
            version of the initial values
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test losses
        """
        if iterations <= 0:
            raise ValueError('number of iterations must be greater than 0')

        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        n_spatial_points = np.prod(cp.mesh.shape(self._vertex_oriented)) \
            if diff_eq.x_dimension else 1

        t = discretize_time_domain(ivp.t_interval, self._d_t)
        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)

        x = np.tile(self._create_input_batch(cp, t[:-1]), (iterations, 1))
        y = np.empty((x.shape[0], diff_eq.y_dimension))
        for epoch in range(iterations):
            offset = epoch * n_spatial_points * (len(t) - 1)
            y_i = y_0

            for i, t_i in enumerate(t[:-1]):
                perturbed_y_i = perturbation_function(t_i, y_i)
                if perturbed_y_i.shape != y_i.shape:
                    raise ValueError(
                        f'perturbed y shape {perturbed_y_i.shape} must match '
                        f'input y shape {y_i.shape}')

                y_i = perturbed_y_i
                t_offset = offset + i * n_spatial_points
                x[
                    t_offset:t_offset + n_spatial_points,
                    diff_eq.x_dimension + 1:
                ] = y_i.reshape((1, -1))

                sub_ivp = InitialValueProblem(
                    cp,
                    (t_i, t_i + self._d_t),
                    DiscreteInitialCondition(cp, y_i, self._vertex_oriented))
                solution = oracle.solve(sub_ivp)
                y_i = solution.discrete_y(self._vertex_oriented)[-1, ...]
                y[t_offset:t_offset + n_spatial_points, :] = \
                    y_i.reshape((-1, diff_eq.y_dimension))

        train_score, test_score = \
            self._train_model(model, x, y, test_size, score_func)
        self._model = model

        return train_score, test_score

    def _create_input_placeholder(
            self,
            cp: ConstrainedProblem) -> np.ndarray:
        """
        Creates a placeholder array for the model inputs. If the constrained
        problem is a PDE, it pre-populates the first x_dimension columns
        corresponding to the spatial coordinates.

        :param cp: the constrained problem to base the inputs on
        :return: the placeholder array for the model inputs
        """
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            return np.empty((1, 1 + diff_eq.y_dimension))

        x = cp.mesh.all_index_coordinates(self._vertex_oriented, flatten=True)
        t = np.empty((len(x), 1))
        y = np.empty((len(x), diff_eq.y_dimension * len(x)))
        return np.hstack([x, t, y])

    def _create_input_batch(
            self,
            cp: ConstrainedProblem,
            t: np.ndarray) -> np.ndarray:
        """
        Creates a 2D array of inputs with a shape of
        (n_mesh_points * n_time_points, x_dimension + 1).

        :param cp: the constrained problem to base the inputs on
        :param t: the discretized time domain of the IVP to create inputs for
        :return: a batch of all inputs
        """
        input_placeholder = self._create_input_placeholder(cp)
        n_mesh_points = input_placeholder.shape[0]

        batch = np.tile(input_placeholder, (len(t), 1))
        batch[:, cp.differential_equation.x_dimension] = \
            np.repeat(t, n_mesh_points)
        return batch

    @staticmethod
    def _train_model(
            model: RegressionModel,
            x: np.ndarray,
            y: np.ndarray,
            test_size: float,
            score_func: Callable[[np.ndarray, np.ndarray], float]
    ) -> Tuple[float, float]:
        """
        Fits the regression model to the training share of the provided data
        points using random splitting and it returns the loss of the model
        evaluated on both the training and test data sets.

        :param model: the regression model to train
        :param x: the inputs
        :param y: the target outputs
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test losses
        """
        if not 0. <= test_size < 1.:
            raise ValueError(
                f'test size ({test_size}) must be between 0 and 1')
        train_size = 1. - test_size

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=train_size,
            test_size=test_size)

        model.fit(x_train, y_train)

        y_train_hat = model.predict(x_train)
        y_test_hat = model.predict(x_test)
        train_score = score_func(y_train, y_train_hat)
        test_score = score_func(y_test, y_test_hat)
        return train_score, test_score
