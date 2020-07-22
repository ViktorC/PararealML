from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

import numpy as np
from deepxde import Model as PINNModel, IC
from deepxde.boundary_conditions import BC
from deepxde.data import TimePDE, PDE
from deepxde.maps.map import Map
from fipy import Solver
from scipy.integrate import solve_ivp, OdeSolver
from sklearn.base import RegressorMixin
from sklearn.utils import all_estimators
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differentiator import Differentiator
from src.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from src.core.integrator import Integrator

SKLearnRegressionModel = Union[
    tuple([_class for name, _class in all_estimators()
           if issubclass(_class, RegressorMixin)])
]
RegressionModel = Union[SKLearnRegressionModel, KerasRegressor]


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
    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        """
        Returns a discretised approximation of the IVP's solution.

        :param ivp: the initial value problem to solve
        :return: the discretised solution of the IVP
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
            d_t: float):
        """
        :param method: the ODE solver to use
        :param d_t: the temporal step size to use
        """
        assert d_t > 0.
        self._method = method
        self._d_t = d_t

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return None

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        assert diff_eq.x_dimension == 0

        t_interval = ivp.t_interval
        time_steps = self._discretise_time_domain(t_interval, self._d_t)
        adjusted_t_interval = (time_steps[0], time_steps[-1])

        result = solve_ivp(
            diff_eq.d_y_over_d_t,
            adjusted_t_interval,
            ivp.initial_condition.discrete_y_0(True),
            self._method,
            time_steps[1:])
        y = np.ascontiguousarray(result.y.T)
        return y


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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
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

        time_steps = self._discretise_time_domain(
            ivp.t_interval, self._d_t)[:-1]

        y = np.empty((len(time_steps),) + bvp.y_vertices_shape)
        y_i = ivp.initial_condition.discrete_y_0(True)

        for i, t_i in enumerate(time_steps):
            y_i = self._integrator.integral(
                y_i,
                t_i,
                self._d_t,
                d_y_over_d_t,
                y_constraints)
            y[i] = y_i

        return y


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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
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

        time_steps = self._discretise_time_domain(
            ivp.t_interval, self._d_t)[:-1]

        y = np.empty((len(time_steps),) + bvp.y_cells_shape)
        for i, t_i in enumerate(time_steps):
            for fipy_var in fipy_vars:
                fipy_var.updateOld()
            for j, fipy_var in enumerate(fipy_vars):
                fipy_terms[j].solve(
                    var=fipy_var,
                    dt=self._d_t,
                    solver=self._solver)
                y[i, ..., j] = fipy_var.value.reshape(mesh_shape)

        return y


class MLOperator(Operator, ABC):
    """
    A base class for machine learning accelerated operators for solving
    differential equations
    """

    def __init__(
            self,
            d_t: float,
            vertex_oriented: bool,
            batch_mode: bool = True):
        """
        :param d_t: the temporal step size to use
        :param vertex_oriented:
        :param batch_mode: whether the operator is to perform a single
        prediction to evaluate the solution at all coordinates using input
        batching; this can be very memory intensive depending on the temporal
        step size
        """
        assert d_t > 0.
        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._batch_mode = batch_mode
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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        assert self._model is not None

        bvp = ivp.boundary_value_problem

        x = self._create_input_placeholder(bvp)
        time_steps = self._discretise_time_domain(
            ivp.t_interval, self._d_t)[1:]

        y_shape = bvp.y_shape(self._vertex_oriented)
        all_y_shape = (len(time_steps),) + y_shape

        if self._batch_mode:
            x_batch = self._create_input_batch(x, time_steps)
            y_hat_batch = self._model.predict(x_batch)
            y = y_hat_batch \
                .reshape(all_y_shape) \
                .astype(np.float, casting='safe')
        else:
            y = np.empty(all_y_shape)
            for i, t_i in enumerate(time_steps):
                x[:, -1] = t_i
                y_hat = self._model.predict(x)
                y[i, ...] = y_hat.reshape(y_shape)

        return y

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

        :param bvp: the boundary value problem to base the
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

    @staticmethod
    def _create_input_batch(
            input_placeholder: np.ndarray,
            discretised_time_domain: np.ndarray
    ) -> np.ndarray:
        """
        Creates a 2D array of inputs with a shape of
        (n_mesh_points * n_time_points, x_dimension + 1).

        :param input_placeholder: the placeholder array for the inputs
        :param discretised_time_domain: the discretised time domain of the IVP
        to create inputs for
        :return: a batch of all inputs
        """
        n_mesh_points = input_placeholder.shape[0]

        x = np.tile(input_placeholder, (len(discretised_time_domain), 1))
        t = np.repeat(discretised_time_domain, n_mesh_points)
        x[:, -1] = t

        return x


class RegressionOperator(MLOperator):
    """
    A machine learning accelerated operator that uses a regression model to
    solve differential equations.
    """

    def train(
            self,
            ivp: InitialValueProblem,
            oracle: Operator,
            model: RegressionModel):
        """
        Fits a regression model to training data generated by the solving the
        provided IVP using the oracle. It keeps the fitted model for use by the
        operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        """
        t = self._discretise_time_domain(ivp.t_interval, oracle.d_t)
        x = self._create_input_placeholder(ivp.boundary_value_problem)
        x_batch = self._create_input_batch(x, t)

        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)
        y_0 = y_0.reshape((1,) + y_0.shape)
        y_batch = np.concatenate((y_0, oracle.trace(ivp)), axis=0)
        y_batch = y_batch.reshape((-1, y_batch.shape[-1]))

        # TODO: shuffle data

        model.fit(x_batch, y_batch)

        self._model = model


class PINNOperator(MLOperator):
    """
    A physics informed neural network (PINN) based differential equation solver
    using the DeepXDE library.
    """

    def train(
            self,
            ivp: InitialValueProblem,
            network: Map,
            training_config: Dict[str, Any]):
        """
        Trains a PINN model on the provided IVP and keeps it for use by the
        operator.

        :param ivp: the IVP to train the PINN on
        :param network: the PINN to use
        :param training_config: a dictionary of training configurations
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
        self._model.train(epochs=n_epochs, batch_size=batch_size)

        scipy_optimiser = training_config.get('scipy_optimiser', None)
        if scipy_optimiser is not None:
            self._model.compile(scipy_optimiser)
            self._model.train()
