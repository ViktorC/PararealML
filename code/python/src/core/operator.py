from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import numpy as np
from deepxde import Model
from deepxde.data import TimePDE, PDE
from deepxde.maps.map import Map

from src.core.differentiator import Differentiator
from src.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from src.core.integrator import Integrator


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
        Returns a discretisation of the the interval [t_a, t_b^) using the
        temporal step size of the operator d_t, where t_b^ is t_b rounded to
        the nearest multiple of d_t.

        :param t: the time interval to discretise
        :return: the array containing the discretised temporal domain
        """
        d_t = self.d_t
        adjusted_t_1 = d_t * round(t[1] / d_t)
        return np.arange(t[0], adjusted_t_1, d_t)


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

        time_steps = self._discretise_time_domain(ivp.t_interval)

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
    A finite volume method based conventional differential equation solver
    using the FiPy library.
    """

    def __init__(
            self,
            d_t: float):
        """
        :param d_t: the temporal step size to use
        """
        assert d_t > 0.
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

        fipy_diff_eq = diff_eq.fipy_equation

        time_steps = self._discretise_time_domain(ivp.t_interval)

        y = np.empty((len(time_steps),) + bvp.y_shape)
        for i, t_i in enumerate(time_steps):
            for j in range(diff_eq.y_dimension):
                y_var_j = fipy_vars[j]
                fipy_diff_eq.solve(var=y_var_j, dt=self._d_t)
                y[i, ..., j] = y_var_j.value.reshape(mesh.shape)

        return y


class PINNOperator(Operator):
    """
    A physics informed neural network (PINN) based differential equation solver
    using the DeepXDE library.
    """

    def __init__(self, d_t: float):
        """
        :param d_t: the temporal step size to use
        """
        assert d_t > 0.
        self._d_t = d_t
        self._model: Optional[Model] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        assert self._model is not None

        bvp = ivp.boundary_value_problem
        diff_eq = bvp.differential_equation

        assert 1 <= diff_eq.x_dimension <= 3

        mesh = bvp.mesh
        mesh_shape = mesh.shape
        n_points = np.prod(mesh_shape)
        x = np.empty((n_points, diff_eq.x_dimension + 1))
        for row_ind, index in enumerate(np.ndindex(mesh_shape)):
            x[row_ind, :-1] = mesh.x(index)

        time_steps = self._discretise_time_domain(ivp.t_interval)

        y_shape = bvp.y_shape
        y = np.empty((len(time_steps),) + bvp.y_shape)
        for i, t_i in enumerate(time_steps):
            x[:, -1] = t_i
            y_hat = self._model.predict(x)
            y[i, ...] = y_hat.reshape(y_shape)

        return y

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

        assert 1 <= diff_eq.x_dimension <= 3

        deepxde_diff_eq = diff_eq.deepxde_equation
        initial_conditions = ivp.deepxde_initial_conditions

        n_domain = training_config['n_domain']
        n_initial = training_config['n_initial']
        sample_distribution = training_config.get(
            'sample_distribution', 'random')
        solution_function = training_config.get('solution_function', None)
        n_test = training_config.get('n_test', None)

        if diff_eq.x_dimension:
            boundary_conditions = ivp.deepxde_boundary_conditions
            n_boundary = training_config['n_boundary']
            ic_bcs: List[Any] = list(initial_conditions)
            ic_bcs += list(boundary_conditions)
            data = TimePDE(
                geometryxtime=ivp.deepxde_geometry_time_domain,
                pde=deepxde_diff_eq,
                ic_bcs=ic_bcs,
                num_domain=n_domain,
                num_boundary=n_boundary,
                num_initial=n_initial,
                train_distribution=sample_distribution,
                solution=solution_function,
                num_test=n_test)
        else:
            data = PDE(
                geometry=ivp.deepxde_time_domain,
                pde=deepxde_diff_eq,
                bcs=initial_conditions,
                num_domain=n_domain,
                num_boundary=n_initial,
                train_distribution=sample_distribution,
                solution=solution_function,
                num_test=n_test)

        self._model = Model(data, network)

        optimiser = training_config.get('optimiser', 'adam')
        learning_rate = training_config.get('learning_rate', .001)
        self._model.compile(optimizer=optimiser, lr=learning_rate)

        n_epochs = training_config['n_epochs']
        batch_size = training_config.get('batch_size', None)
        self._model.train(epochs=n_epochs, batch_size=batch_size)
