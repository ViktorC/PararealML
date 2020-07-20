from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

import numpy as np
from deepxde import Model, IC
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

SKLearnRegressor = Union[
    tuple([_class for name, _class in all_estimators()
           if issubclass(_class, RegressorMixin)])
]
Regressor = Union[SKLearnRegressor, KerasRegressor]


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

    @abstractmethod
    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        """
        Returns a discretised approximation of the IVP's solution.

        :param ivp: the initial value problem to solve
        :return: the discretised solution of the IVP
        """

    def _discretise_time_domain(self, t: TemporalDomainInterval) -> np.ndarray:
        """
        Returns a discretisation of the interval [t_a, t_b^) using the temporal
        step size of the operator d_t, where t_b^ = t_a + n * d_t and n E Z,
        n = argmin |t_b^ - t_b|.

        :param t: the time interval to discretise
        :return: the array containing the discretised temporal domain
        """
        d_t = self.d_t
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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        assert diff_eq.x_dimension == 0

        t_interval = ivp.t_interval
        time_steps = self._discretise_time_domain(t_interval)
        adjusted_t_interval = (time_steps[0], time_steps[-1])

        result = solve_ivp(
            diff_eq.d_y_over_d_t,
            adjusted_t_interval,
            ivp.initial_condition.discrete_y_0,
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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation
        d_x = bvp.mesh.d_x if diff_eq.x_dimension else None
        y_constraints = bvp.y_constraints
        d_y_boundary_constraints = bvp.d_y_boundary_constraints

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return diff_eq.d_y_over_d_t(
                _t,
                _y,
                d_x,
                self._differentiator,
                d_y_boundary_constraints,
                y_constraints)

        time_steps = self._discretise_time_domain(ivp.t_interval)[:-1]

        y = np.empty((len(time_steps),) + bvp.y_shape)
        y_i = ivp.initial_condition.discrete_y_0

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

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation
        mesh = bvp.mesh

        assert 1 <= diff_eq.x_dimension <= 3

        y_0 = ivp.initial_condition.discrete_y_0

        fipy_vars = bvp.fipy_vars
        for i, fipy_var in enumerate(fipy_vars):
            fipy_var.setValue(value=y_0[..., i].flatten())

        fipy_terms = diff_eq.fipy_terms(fipy_vars)

        time_steps = self._discretise_time_domain(ivp.t_interval)[:-1]

        y = np.empty((len(time_steps),) + bvp.y_shape)
        for i, t_i in enumerate(time_steps):
            for fipy_var in fipy_vars:
                fipy_var.updateOld()
            for j, fipy_var in enumerate(fipy_vars):
                fipy_terms[j].solve(
                    var=fipy_var,
                    dt=self._d_t,
                    solver=self._solver)
                y[i, ..., j] = fipy_var.value.reshape(mesh.shape)

        return y


class MLOperator(Operator, ABC):
    """
    A base class for machine learning accelerated operators for solving
    differential equations
    """

    def __init__(self, d_t: float):
        """
        :param d_t: the temporal step size to use
        """
        assert d_t > 0.
        self._d_t = d_t
        self._model: Optional[Union[Regressor, Model]] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def model(self) -> Optional[Union[Regressor, Model]]:
        return self._model

    @model.setter
    def model(self, model: Optional[Union[Regressor, Model]]):
        self._model = model

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        assert self._model is not None

        bvp = ivp.boundary_value_problem

        x = self._create_input_placeholder(bvp)
        time_steps = self._discretise_time_domain(ivp.t_interval)[1:]

        y_shape = bvp.y_shape
        y = np.empty((len(time_steps),) + bvp.y_shape)
        for i, t_i in enumerate(time_steps):
            x[:, -1] = t_i
            y_hat = self._model.predict(x)
            y[i, ...] = y_hat.reshape(y_shape)

        return y

    @staticmethod
    def _create_input_placeholder(bvp: BoundaryValueProblem) -> np.ndarray:
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
            mesh_shape = mesh.shape
            n_points = np.prod(mesh_shape)
            x = np.empty((n_points, diff_eq.x_dimension + 1))
            for row_ind, index in enumerate(np.ndindex(mesh_shape)):
                x[row_ind, :-1] = mesh.x(index)
        else:
            x = np.empty((1, 1))

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
            model: Regressor):
        """
        Fits a regression model to training data generated by the solving the
        provided IVP using the oracle. It keeps the fitted model for use by the
        operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        """
        x = self._create_input_placeholder(ivp.boundary_value_problem)
        t = self._discretise_time_domain(ivp.t_interval)

        x = np.repeat(x, len(t), axis=0)
        t = np.repeat(t, np.prod(ivp.boundary_value_problem.mesh.shape))
        x[:, -1] = t

        y_0 = ivp.initial_condition.discrete_y_0.reshape(
            1, *ivp.boundary_value_problem.y_shape)
        y = np.concatenate((y_0, oracle.trace(ivp)), axis=0)
        y = y.reshape((-1, y.shape[-1]))

        model.fit(x, y)

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

        self._model = Model(data, network)

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
