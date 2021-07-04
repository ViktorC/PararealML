from typing import List, Optional, Union, Tuple, Callable, Sequence, Dict

import numpy as np
import sympy as sp
import tensorflow as tf
from deepxde import Model as PINNModel, IC
from deepxde.boundary_conditions import BC, DirichletBC, NeumannBC
from deepxde.data import TimePDE, PDE
from deepxde.geometry import TimeDomain, GeometryXTime, Interval, Rectangle, \
    Cuboid
from deepxde.geometry.geometry import Geometry
from deepxde.maps.map import Map
from deepxde.model import TrainState, LossHistory
from tensorflow import Tensor

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import LhsType, DifferentialEquation
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.mesh import Mesh
from pararealml.core.operators.ml_operator import MLOperator


class DeepONetOperator(MLOperator):
    """
    A physics informed DeepONet based unsupervised machine learning operator
    for solving initial value problems using the DeepXDE library.
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
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        if diff_eq.x_dimension > 3:
            raise ValueError

        symbol_set = set()
        symbolic_equation_system = diff_eq.symbolic_equation_system
        for rhs_element in symbolic_equation_system.rhs:
            symbol_set.update(rhs_element.free_symbols)

        symbol_map = self._create_symbol_map(ivp.constrained_problem)
        symbol_arg_funcs = [symbol_map[symbol] for symbol in symbol_set]

        rhs_lambda = sp.lambdify(
            [symbol_set],
            symbolic_equation_system.rhs,
            'numpy')

        lhs_functions = self._create_lhs_functions(diff_eq)

        def diff_eq_error(
                x: Tensor,
                y: Tensor
        ) -> Sequence[Tensor]:
            rhs = rhs_lambda(
                [func(x, y) for func in symbol_arg_funcs]
            )
            return [
                lhs_functions[j](x, y) - rhs[j]
                for j in range(diff_eq.y_dimension)
            ]

        n_domain = training_config['n_domain']
        n_initial = training_config['n_initial']
        n_test = training_config.get('n_test', None)
        sample_distribution = training_config.get(
            'sample_distribution', 'random')
        solution_function = training_config.get('solution_function', None)

        geometry = self._create_deepxde_geometry(cp.mesh) \
            if diff_eq.x_dimension else None
        initial_conditions = self._create_deepxde_initial_conditions(
            ivp, geometry)

        if diff_eq.x_dimension:
            boundary_conditions = self._create_deepxde_boundary_conditions(
                ivp, geometry)
            n_boundary = training_config['n_boundary']
            ic_bcs: List[Union[IC, BC]] = list(initial_conditions)
            ic_bcs += list(boundary_conditions)
            data = TimePDE(
                geometryxtime=GeometryXTime(
                    geometry,
                    TimeDomain(*ivp.t_interval)),
                pde=diff_eq_error,
                ic_bcs=ic_bcs,
                num_domain=n_domain,
                num_boundary=n_boundary,
                num_initial=n_initial,
                num_test=n_test,
                train_distribution=sample_distribution,
                solution=solution_function)
        else:
            data = PDE(
                geometry=TimeDomain(*ivp.t_interval),
                pde=diff_eq_error,
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

    @staticmethod
    def _create_deepxde_geometry(mesh: Mesh) -> Geometry:
        """
        Creates and returns the DeepXDE equivalent of the spatial domain
        represented by the provided mesh.
        """
        x_intervals = mesh.x_intervals
        x_dimension = len(x_intervals)
        if x_dimension == 1:
            x_interval = x_intervals[0]
            geometry = Interval(*x_interval)
        elif x_dimension == 2:
            geometry = Rectangle(*zip(*x_intervals))
        elif x_dimension == 3:
            geometry = Cuboid(*zip(*x_intervals))
        else:
            raise NotImplementedError

        return geometry

    @staticmethod
    def _create_deepxde_initial_conditions(
            ivp: InitialValueProblem,
            geometry: Geometry
    ) -> Sequence[IC]:
        """
        Creates the DeepXDE equivalent of the initial condition.

        :param ivp: the initial value problem to create DeepXDE initial
            conditions for
        :param geometry: the DeepXDE equivalent of the IVP's mesh
        :return: a sequence of the DeepXDE initial conditions
        """
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        time_domain = TimeDomain(*ivp.t_interval)
        geometry_time_domain = GeometryXTime(geometry, time_domain) \
            if diff_eq.x_dimension else time_domain

        condition_functions = DeepONetOperator._create_deepxde_condition_functions(
            cp, ivp.initial_condition.y_0)

        return [
            IC(
                geometry_time_domain,
                cond_func,
                lambda _, on_initial: on_initial, y_ind)
            for y_ind, cond_func in enumerate(condition_functions)
        ]

    @staticmethod
    def _create_deepxde_boundary_conditions(
            ivp: InitialValueProblem,
            geometry: Geometry
    ) -> Optional[Sequence[BC]]:
        """
        Creates the DeepXDE equivalent of the boundary conditions.

        :param ivp: the initial value problem to create DeepXDE boundary
            conditions for
        :param geometry: the DeepXDE equivalent of the IVP's mesh
        :return: a sequence of the DeepXDE boundary conditions or None if the
            IVP is based on an ODE
        """
        cp = ivp.constrained_problem
        if not cp.differential_equation.x_dimension:
            return None

        boundary_conditions: List[BC] = []

        for axis, bc_pair in enumerate(cp.boundary_conditions):
            if bc_pair is not None:
                for bc_ind, bc in enumerate(bc_pair):
                    boundary_value = cp.mesh.x_intervals[axis][bc_ind]

                    DeepONetOperator._add_deepxde_boundary_conditions_for_all_y(
                        ivp,
                        geometry,
                        bc.has_y_condition,
                        bc.y_condition,
                        DirichletBC,
                        axis,
                        boundary_value,
                        boundary_conditions)
                    DeepONetOperator._add_deepxde_boundary_conditions_for_all_y(
                        ivp,
                        geometry,
                        bc.has_d_y_condition,
                        bc.d_y_condition,
                        NeumannBC,
                        axis,
                        boundary_value,
                        boundary_conditions)

        return boundary_conditions

    @staticmethod
    def _add_deepxde_boundary_conditions_for_all_y(
            ivp: InitialValueProblem,
            geometry: Geometry,
            has_condition: bool,
            condition_function: Callable[
                [Sequence[float], Optional[float]],
                Optional[Sequence[Optional[float]]]
            ],
            deepxde_boundary_condition_type: type,
            fixed_axis: int,
            boundary_value: float,
            boundary_conditions: List[BC]):
        """
        Creates a DeepXDE boundary condition for each element of y and appends
        them to the list of boundary conditions.

        :param ivp: the initial value problem to create condition functions for
        :param geometry: the DeepXDE equivalent of the IVP's mesh
        :param has_condition: whether there is an organic boundary condition
            specified
        :param condition_function: the organic boundary condition
        :param deepxde_boundary_condition_type: the DeepXDE equivalent of the
            type of the organic boundary condition
        :param fixed_axis: the axis normal to the boundary
        :param boundary_value: the value along the fixed axis at the boundary
        :param boundary_conditions: the list of DeepXDE boundary conditions to
            append the created boundary conditions to
        """
        if has_condition:
            cp = ivp.constrained_problem
            deepxde_condition_functions = \
                DeepONetOperator._create_deepxde_condition_functions(
                    cp, condition_function, fixed_axis)
            deepxde_geometry_time_domain = GeometryXTime(
                geometry,
                TimeDomain(*ivp.t_interval))

            for y_ind, cond_func in \
                    enumerate(deepxde_condition_functions):
                def predicate(
                        x: np.ndarray,
                        on_boundary: bool,
                        _y_ind: int = y_ind) -> bool:
                    return on_boundary \
                           and np.isclose(x[fixed_axis], boundary_value) \
                           and (condition_function(x[:-1], x[-1])[_y_ind]
                                is not None)

                boundary_conditions.append(
                    deepxde_boundary_condition_type(
                        deepxde_geometry_time_domain,
                        cond_func,
                        predicate,
                        y_ind))

    @staticmethod
    def _create_deepxde_condition_functions(
            cp: ConstrainedProblem,
            condition_function: Union[
                Callable[[Optional[Sequence[float]]],
                         Optional[Sequence[float]]],
                Callable[[Sequence[float], Optional[float]],
                         Optional[Sequence[Optional[float]]]]],
            fixed_axis: Optional[int] = None
    ) -> Sequence[Callable[[np.ndarray], np.ndarray]]:
        """
        Creates a list of functions that can be used to define DeepXDE boundary
        conditions.

        :param cp: the constrained problem to create condition functions for
        :param condition_function: a condition function in the format of the
            y_0 function of well defined initial conditions or the y_condition
            or d_y_condition functions of boundary conditions
        :param fixed_axis: the fixed axis in case the condition function is
            a boundary condition function
        :return: a list of DeepXDE condition functions with an element for each
            component of the output array of the organic condition function
        """
        deepxde_condition_functions = []
        for y_ind in range(cp.differential_equation.y_dimension):
            def condition(x: np.ndarray, _y_ind: int = y_ind) -> np.ndarray:
                n_rows = x.shape[0]

                if fixed_axis is not None:
                    x = np.delete(x, fixed_axis, axis=1)
                    values = np.array([
                        condition_function(x[i, :-1], x[i, -1])[_y_ind]
                        for i in range(n_rows)
                    ])
                else:
                    values = np.array([
                        condition_function(x[i, :-1])[_y_ind]
                        for i in range(n_rows)
                    ])

                values = values.reshape((n_rows, 1))
                return values

            deepxde_condition_functions.append(condition)

        return deepxde_condition_functions

    @staticmethod
    def _create_lhs_functions(
            diff_eq: DifferentialEquation
    ) -> Sequence[Callable[[Tensor, Tensor], Tensor]]:
        """
        Returns a list of functions for calculating the left hand sides of the
        differential equation given x and y.

        :param diff_eq: the differential equation to compute the left hand
        sides for
        :return: a list of functions
        """
        lhs_functions = []
        for i, lhs_type in \
                enumerate(diff_eq.symbolic_equation_system.lhs_types):
            if lhs_type == LhsType.D_Y_OVER_D_T:
                lhs_functions.append(
                    lambda x, y, _i=i:
                    tf.gradients(y[:, _i:_i + 1], x)[0][:, -1:]
                )
            elif lhs_type == LhsType.Y:
                lhs_functions.append(lambda x, y, _i=i: y[:, _i:_i + 1])
            elif lhs_type == LhsType.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda x, y, _i=i:
                    tf.math.reduce_sum(
                        tf.gradients(
                            tf.gradients(
                                y[:, _i:_i + 1],
                                x
                            )[0][:, :diff_eq.x_dimension],
                            x
                        )[0][:, :diff_eq.x_dimension],
                        -1,
                        True
                    )
                )
            else:
                raise ValueError

        return lhs_functions

    @staticmethod
    def _create_symbol_map(
            cp: ConstrainedProblem
    ) -> Dict[sp.Symbol, Callable[[Tensor, Tensor], Tensor]]:
        """
        Creates a dictionary mapping symbols to functions returning the values
        of these symbols given x and y.

        :param cp: the constrained problem to create a symbol map for
        :return: a dictionary mapping symbols to functions
        """
        diff_eq = cp.differential_equation

        symbol_map = {diff_eq.symbols.t: lambda x, y: x[:, -1:]}

        for i, y_element in enumerate(diff_eq.symbols.y):
            symbol_map[y_element] = lambda x, y, _i=i: y[:, _i:_i + 1]

        if diff_eq.x_dimension:
            y_gradient = diff_eq.symbols.y_gradient
            y_hessian = diff_eq.symbols.y_hessian
            y_laplacian = diff_eq.symbols.y_laplacian
            y_divergence = diff_eq.symbols.y_divergence
            y_curl = diff_eq.symbols.y_curl

            for i in range(diff_eq.y_dimension):
                symbol_map[y_laplacian[i]] = lambda x, y, _i=i: \
                    tf.math.reduce_sum(
                        tf.gradients(
                            tf.gradients(
                                y[:, _i:_i + 1],
                                x
                            )[0][:, :diff_eq.x_dimension],
                            x
                        )[0][:, :diff_eq.x_dimension],
                        -1,
                        True)

                for j in range(diff_eq.x_dimension):
                    symbol_map[y_gradient[i, j]] = lambda x, y, _i=i, _j=j: \
                        tf.gradients(y[:, _i:_i + 1], x)[0][:, _j:_j + 1]

                    for k in range(diff_eq.x_dimension):
                        symbol_map[y_hessian[i, j, k]] = \
                            lambda x, y, _i=i, _j=j, _k=k: \
                            tf.gradients(
                                tf.gradients(
                                    y[:, _i:_i + 1],
                                    x
                                )[0][:, _j:_j + 1],
                                x
                            )[0][:, _k:_k + 1]

            for index in np.ndindex(
                    (diff_eq.y_dimension,) * diff_eq.x_dimension):
                symbol_map[y_divergence[index]] = lambda x, y, _index=index: \
                    tf.math.reduce_sum(
                        tf.stack([
                            tf.gradients(
                                y[:, _index[_i]:_index[_i] + 1],
                                x
                            )[0][:, _i:_i + 1]
                            for _i in range(diff_eq.x_dimension)
                        ]),
                        axis=0)
                if diff_eq.x_dimension == 2:
                    symbol_map[y_curl[index]] = lambda x, y, _index=index: \
                        tf.gradients(
                            y[:, _index[1]:_index[1] + 1], x)[0][:, 0:1] - \
                        tf.gradients(
                            y[:, _index[0]:_index[0] + 1], x)[0][:, 1:2]
                elif diff_eq.x_dimension == 3:
                    symbol_map[y_curl[index]] = lambda x, y, _index=index: \
                        tf.stack([
                            tf.gradients(
                                y[:, _index[2]:_index[2] + 1], x)[0][:, 1:2] -
                            tf.gradients(
                                y[:, _index[1]:_index[1] + 1], x)[0][:, 2:3],
                            tf.gradients(
                                y[:, _index[0]:_index[0] + 1], x)[0][:, 2:3] -
                            tf.gradients(
                                y[:, _index[2]:_index[2] + 1], x)[0][:, 0:1],
                            tf.gradients(
                                y[:, _index[1]:_index[1] + 1], x)[0][:, 0:1] -
                            tf.gradients(
                                y[:, _index[0]:_index[0] + 1], x)[0][:, 1:2]
                        ], axis=-1)

        return symbol_map
