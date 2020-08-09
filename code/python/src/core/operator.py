import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Callable

import numpy as np
from deepxde import Model as PINNModel, IC
from deepxde.boundary_conditions import BC
from deepxde.data import TimePDE, PDE
from deepxde.maps.map import Map
from deepxde.model import TrainState, LossHistory
from fipy import Solver
from scipy.integrate import solve_ivp, OdeSolver

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differentiator import Differentiator
from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from src.core.integrator import Integrator
from src.core.solution import Solution
from src.utils.io import suppress_stdout
from src.utils.ml import train_regression_model, RegressionModel, \
    root_mean_squared_error


class Operator(ABC):
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    @property
    @abstractmethod
    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """

    @property
    @abstractmethod
    def vertex_oriented(self) -> Optional[bool]:
        """
        Returns whether the operator evaluates the solutions at the vertices
        of the spatial mesh or at the cell centers. If the operator is only an
        ODE solver, it can return None.
        """

    @abstractmethod
    def solve(self, ivp: InitialValueProblem) -> Solution:
        """
        Returns the IVP's solution.

        :param ivp: the initial value problem to solve
        :return: the solution of the IVP
        """

    @staticmethod
    def _discretise_time_domain(
            t: TemporalDomainInterval,
            d_t: float
    ) -> np.ndarray:
        """
        Returns a discretisation of the interval [t_a, t_b^) using the provided
        temporal step size d_t, where t_b^ = t_a + n * d_t and n E Z,
        n = argmin |t_b^ - t_b|.

        :param t: the time interval to discretise
        :param d_t: the temporal step size
        :return: the array containing the discretised temporal domain
        """
        t_0 = t[0]
        steps = round((t[1] - t_0) / d_t)
        t_1 = t_0 + steps * d_t
        return np.linspace(t_0, t_1, steps + 1)


class ODEOperator(Operator):
    """
    An ordinary differential equation solver using the SciPy library.
    """

    def __init__(
            self,
            method: Union[str, OdeSolver],
            d_t: float,
            atol: float = 1e-6,
            rtol: float = 1e-3):
        """
        :param method: the ODE solver to use
        :param d_t: the temporal step size to use
        :param atol: the absolute error tolerance
        :param rtol: the relative error tolerance
        """
        assert d_t > 0.
        assert atol > 0.
        assert rtol > 0.

        self._method = method
        self._d_t = d_t
        self._atol = atol
        self._rtol = rtol

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return None

    def solve(self, ivp: InitialValueProblem) -> Solution:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        assert diff_eq.x_dimension == 0

        t_interval = ivp.t_interval
        time_points = self._discretise_time_domain(t_interval, self._d_t)
        adjusted_t_interval = (time_points[0], time_points[-1])

        result = solve_ivp(
            diff_eq.d_y_over_d_t,
            adjusted_t_interval,
            ivp.initial_condition.discrete_y_0(),
            self._method,
            time_points[1:],
            atol=self._atol,
            rtol=self._rtol,
            dense_output=False,
            vectorized=False)

        if not result.success:
            raise ValueError(
                f'status code: {result.status}, message: {result.message}')

        y = np.ascontiguousarray(result.y.T)
        return Solution(bvp, time_points[1:], y, d_t=self._d_t)


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
        assert d_t > 0.
        self._integrator = integrator
        self._differentiator = differentiator
        self._d_t = d_t

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return True

    def solve(self, ivp: InitialValueProblem) -> Solution:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation
        d_x = bvp.mesh.d_x if diff_eq.x_dimension else None
        y_constraints = bvp.y_vertex_constraints
        d_y_boundary_constraints = bvp.d_y_boundary_vertex_constraints

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return diff_eq.d_y_over_d_t(
                _t,
                _y,
                d_x,
                self._differentiator,
                d_y_boundary_constraints,
                y_constraints)

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)

        y = np.empty((len(time_points) - 1,) + bvp.y_vertices_shape)
        y_i = ivp.initial_condition.discrete_y_0(True)

        for i, t_i in enumerate(time_points[:-1]):
            y_i = self._integrator.integral(
                y_i,
                t_i,
                self._d_t,
                d_y_over_d_t,
                y_constraints)
            y[i] = y_i

        return Solution(
            bvp, time_points[1:], y, vertex_oriented=True, d_t=self._d_t)


class FVMOperator(Operator):
    """
    A finite volume method based conventional partial differential equation
    solver using the FiPy library.
    """

    def __init__(
            self,
            solver: Solver,
            d_t: float):
        """
        :param solver: the FiPy solver to use
        :param d_t: the temporal step size to use
        """
        assert d_t > 0.
        self._solver = solver
        self._d_t = d_t

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return False

    def solve(self, ivp: InitialValueProblem) -> Solution:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        assert 1 <= diff_eq.x_dimension <= 3

        mesh = bvp.mesh
        mesh_shape = mesh.shape(False)
        y_0 = ivp.initial_condition.discrete_y_0(False)

        fipy_vars = bvp.fipy_vars
        for i, fipy_var in enumerate(fipy_vars):
            fipy_var.setValue(value=y_0[..., i].flatten())

        fipy_terms = diff_eq.fipy_terms(fipy_vars)

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)

        y = np.empty((len(time_points) - 1,) + bvp.y_cells_shape)
        for i, t_i in enumerate(time_points[:-1]):
            for fipy_var in fipy_vars:
                fipy_var.updateOld()

            for j, fipy_var in enumerate(fipy_vars):
                fipy_terms[j].solve(
                    var=fipy_var,
                    dt=self._d_t,
                    solver=self._solver)
                y[i, ..., j] = fipy_var.value.reshape(mesh_shape)

        return Solution(
            bvp, time_points[1:], y, vertex_oriented=False, d_t=self._d_t)


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
        assert d_t > 0.
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

    @abstractmethod
    def model_input_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        """
        Returns the shape of the input of the model for the provided IVP.

        :param ivp: the initial value problem to solve
        :return: the expected input shape
        """

    @abstractmethod
    def model_output_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        """
        Returns the shape of the output of the model for the provided IVP.

        :param ivp: the initial value problem to solve
        :return: the expected output shape
        """

    def _create_input_placeholder(
            self,
            bvp: BoundaryValueProblem
    ) -> np.ndarray:
        """
        Creates a placeholder array for the ML model inputs. If the BVP is an
        ODE, it returns an empty array of shape (1, 1) into which t can be
        substituted to create x. If the BVP is a PDE, it returns an array of
        shape (n_mesh_points, x_dimension + 1) whose each row is populated with
        the spatial coordinates of the corresponding mesh point in addition to
        an empty column for t.

        :param bvp: the boundary value problem to base the inputs on
        :return: the placeholder array for the ML inputs
        """
        diff_eq = bvp.differential_equation

        if diff_eq.x_dimension:
            mesh = bvp.mesh
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
            bvp: BoundaryValueProblem,
            time_points: np.ndarray
    ) -> np.ndarray:
        """
        Creates a 2D array of inputs with a shape of
        (n_mesh_points * n_time_points, x_dimension + 1).

        :param bvp: the boundary value problem to base the inputs on
        :param time_points: the discretised time domain of the IVP to create
            inputs for
        :return: a batch of all inputs
        """
        input_placeholder = self._create_input_placeholder(bvp)
        n_mesh_points = input_placeholder.shape[0]

        x = np.tile(input_placeholder, (len(time_points), 1))
        t = np.repeat(time_points, n_mesh_points)
        x[:, bvp.differential_equation.x_dimension] = t

        return x


class StatelessMLOperator(MLOperator, ABC):
    """
    A base class for machine learning operators modelling the solution of
    initial value problems.
    """

    def __init__(
            self,
            d_t: float,
            vertex_oriented: bool,
            batch_mode: bool = True):
        """
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        :param batch_mode: whether the operator is to perform a single
            prediction to evaluate the solution at all coordinates using input
            batching; this can be very memory intensive depending on the
            temporal step size
        """
        super(StatelessMLOperator, self).__init__(d_t, vertex_oriented)
        self._batch_mode = batch_mode

    def model_input_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        diff_eq = ivp.boundary_value_problem.differential_equation
        return diff_eq.x_dimension + 1,

    def model_output_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        diff_eq = ivp.boundary_value_problem.differential_equation
        return diff_eq.y_dimension,

    @suppress_stdout
    def solve(self, ivp: InitialValueProblem) -> Solution:
        assert self._model is not None

        bvp = ivp.boundary_value_problem

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)[1:]

        y_shape = bvp.y_shape(self._vertex_oriented)
        all_y_shape = (len(time_points),) + y_shape

        if self._batch_mode:
            x_batch = self._create_input_batch(bvp, time_points)
            y_hat_batch = self._model.predict(x_batch)
            y = np.ascontiguousarray(
                y_hat_batch.reshape(all_y_shape),
                dtype=np.float)
        else:
            x = self._create_input_placeholder(bvp)
            y = np.empty(all_y_shape)
            for i, t_i in enumerate(time_points):
                x[:, -1] = t_i
                y_hat = self._model.predict(x)
                y[i, ...] = y_hat.reshape(y_shape)

        return Solution(
            bvp,
            time_points,
            y, vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)


class StatefulMLOperator(MLOperator, ABC):
    """
    A base class for machine learning operators that model the solution of an
    initial value problem at the next time step given its solution at the
    current time step.
    """

    def model_input_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        diff_eq = ivp.boundary_value_problem.differential_equation
        return diff_eq.x_dimension + 1 + diff_eq.y_dimension,

    def model_output_shape(self, ivp: InitialValueProblem) -> Tuple[int]:
        diff_eq = ivp.boundary_value_problem.differential_equation
        return diff_eq.y_dimension,

    @suppress_stdout
    def solve(self, ivp: InitialValueProblem) -> Solution:
        assert self._model is not None

        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)

        y_shape = bvp.y_shape(self._vertex_oriented)

        x = self._create_input_placeholder(bvp)
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
            bvp,
            time_points[1:],
            y,
            vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)


class PINNOperator(StatelessMLOperator):
    """
    A physics informed neural network based unsupervised machine learning
    operator for solving initial value problems using the DeepXDE library.
    """

    def train(
            self,
            ivp: InitialValueProblem,
            network: Map,
            **training_config: Union[int, float, str]
    ) -> Tuple[LossHistory, TrainState]:
        """
        Trains a PINN model on the provided IVP and keeps it for use by the
        operator.

        :param ivp: the IVP to train the PINN on
        :param network: the PINN to use
        :param training_config: keyworded training configuration arguments
        :return: a tuple of the loss history and the training state
        """
        diff_eq = ivp.boundary_value_problem.differential_equation

        assert diff_eq.x_dimension <= 3

        deepxde_diff_eq = diff_eq.deepxde_tensors
        initial_conditions = ivp.deepxde_initial_conditions

        n_domain = training_config['n_domain']
        n_initial = training_config['n_initial']
        n_test = training_config.get('n_test', None)
        sample_distribution = training_config.get(
            'sample_distribution', 'random')
        solution_function = training_config.get('solution_function', None)

        if diff_eq.x_dimension:
            boundary_conditions = ivp.deepxde_boundary_conditions
            n_boundary = training_config['n_boundary']
            ic_bcs: List[Union[IC, BC]] = list(initial_conditions)
            ic_bcs += list(boundary_conditions)
            data = TimePDE(
                geometryxtime=ivp.deepxde_geometry_time_domain,
                pde=deepxde_diff_eq,
                ic_bcs=ic_bcs,
                num_domain=n_domain,
                num_boundary=n_boundary,
                num_initial=n_initial,
                num_test=n_test,
                train_distribution=sample_distribution,
                solution=solution_function)
        else:
            data = PDE(
                geometry=ivp.deepxde_time_domain,
                pde=deepxde_diff_eq,
                bcs=initial_conditions,
                num_domain=n_domain,
                num_boundary=n_initial,
                num_test=n_test,
                train_distribution=sample_distribution,
                solution=solution_function)

        self._model = PINNModel(data, network)

        optimiser = training_config['optimiser']
        learning_rate = training_config.get('learning_rate', None)
        self._model.compile(optimizer=optimiser, lr=learning_rate)

        n_epochs = training_config['n_epochs']
        batch_size = training_config.get('batch_size', None)
        loss_history, train_state = self._model.train(
            epochs=n_epochs, batch_size=batch_size)

        scipy_optimiser = training_config.get('scipy_optimiser', None)
        if scipy_optimiser is not None:
            self._model.compile(scipy_optimiser)
            loss_history, train_state = self._model.train()

        return loss_history, train_state


class StatelessRegressionOperator(StatelessMLOperator):
    """
    A supervised machine learning operator that uses regression to model the
    solution of initial value problems.
    """

    def train(
            self,
            ivp: InitialValueProblem,
            oracle: Operator,
            model: RegressionModel,
            subsampling_factor: Optional[float] = None,
            test_size: float = .2,
            score_func: Callable[[np.ndarray, np.ndarray], float] =
            root_mean_squared_error
    ) -> Tuple[float, float]:
        """
        Fits a regression model to training data generated by the solving the
        provided IVP using the oracle. It keeps the fitted model for use by the
        operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        :param subsampling_factor: the fraction of all data points that should
            be sampled for training; it has to be greater than 0 and less than
            or equal to 1; if it is None, all data points will be used
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test losses
        """
        assert subsampling_factor is None or 0. < subsampling_factor <= 1.

        time_steps = self._discretise_time_domain(ivp.t_interval, oracle.d_t)
        x_batch = self._create_input_batch(
            ivp.boundary_value_problem,
            time_steps)

        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)
        y_0 = y_0.reshape((1,) + y_0.shape)
        y = oracle.solve(ivp).discrete_y(self._vertex_oriented)
        y_batch = np.concatenate((y_0, y), axis=0)
        y_batch = y_batch.reshape((-1, y_batch.shape[-1]))

        if subsampling_factor is not None and subsampling_factor < 1.:
            n_data_points = x_batch.shape[0]
            indices = np.random.randint(
                0,
                n_data_points,
                size=math.ceil(subsampling_factor * n_data_points))
            x_batch = x_batch[indices, :]
            y_batch = y_batch[indices, :]

        train_score, test_score = train_regression_model(
            model, x_batch, y_batch, test_size, score_func)
        self._model = model

        return train_score, test_score


class StatefulRegressionOperator(StatefulMLOperator):
    """
    A supervised machine learning operator that uses regression to model
    another operator for solving initial value problems.
    """

    def train(
            self,
            ivp: InitialValueProblem,
            oracle: Operator,
            model: RegressionModel,
            iterations: int,
            noise_sd: Union[float, Tuple[float, float]],
            relative_noise: bool = False,
            test_size: float = .2,
            score_func: Callable[[np.ndarray, np.ndarray], float] =
            root_mean_squared_error
    ) -> Tuple[float, float]:
        """
        Fits a regression model to training data generated by the oracle. The
        inputs of the model are spatio-temporal coordinates and the value of
        the solution at the coordinates and its outputs are the value of the
        solution at the next time step. The training data is generated by
        using the oracle to solve sub-IVPs with randomised initial conditions
        and a time domain extent matching the step size of this operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        :param iterations: the number of data generation iterations
        :param noise_sd: the standard deviation of the Gaussian noise to add to
            the initial conditions of the sub-IVPs. It can be either a scalar,
            in which case the noise is sampled from the same distribution for
            each sub-IVP, or a tuple of two values. The first value of the
            tuple is the standard deviation of the distribution from which the
            noise added to the first sub-IVP is sampled and the second value of
            the tuple is the standard deviation of the distribution from which
            the noise added to the last sub-IVP is sampled. The standard
            deviations of the distribution associated with the sub-IVPs in
            between are calculated using linear interpolation.
        :param relative_noise: whether the noise standard deviation is relative
            to the value of the initial conditions of the sub-IVPs
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test losses
        """
        assert iterations > 0

        if isinstance(noise_sd, (tuple, list)):
            assert len(noise_sd) == 2
            assert noise_sd[0] >= 0. and noise_sd[1] >= 0.
        else:
            assert isinstance(noise_sd, float)
            assert noise_sd >= 0.

            noise_sd = (noise_sd, noise_sd)

        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        n_spatial_points = np.prod(bvp.mesh.shape(self._vertex_oriented)) \
            if diff_eq.x_dimension else 1

        time_points = self._discretise_time_domain(ivp.t_interval, self._d_t)
        last_sub_ivp_start_time_point = len(time_points) - 2

        x_batch = self._create_input_batch(bvp, time_points[:-1])
        x_batch = np.concatenate(
            (x_batch, np.empty((x_batch.shape[0], diff_eq.y_dimension))),
            axis=-1)
        all_x = np.tile(x_batch, (iterations, 1))

        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)
        all_y = np.empty((all_x.shape[0], diff_eq.y_dimension))

        for epoch in range(iterations):
            offset = epoch * x_batch.shape[0]
            y_i = y_0

            for i, t_i in enumerate(time_points[:-1]):
                if len(time_points) > 2:
                    interpolated_noise_sd = \
                        (noise_sd[0] * (last_sub_ivp_start_time_point - i) +
                         noise_sd[1] * i) / last_sub_ivp_start_time_point
                else:
                    interpolated_noise_sd = noise_sd[0]
                if relative_noise:
                    interpolated_noise_sd = y_i * interpolated_noise_sd

                y_i += np.random.normal(
                    loc=0.,
                    scale=interpolated_noise_sd,
                    size=y_i.shape).astype(y_i.dtype)

                time_point_offset = offset + i * n_spatial_points
                all_x[time_point_offset:time_point_offset + n_spatial_points,
                      -diff_eq.y_dimension:] = \
                    y_i.reshape((-1, diff_eq.y_dimension))

                sub_ivp = InitialValueProblem(
                    bvp,
                    (t_i, t_i + self._d_t),
                    DiscreteInitialCondition(bvp, y_i, self._vertex_oriented))
                solution = oracle.solve(sub_ivp)

                y_i = solution.discrete_y(self._vertex_oriented)[-1, ...]
                all_y[time_point_offset:time_point_offset + n_spatial_points,
                      :] = y_i.reshape((-1, diff_eq.y_dimension))

        train_score, test_score = train_regression_model(
            model, all_x, all_y, test_size, score_func)
        self._model = model

        return train_score, test_score
