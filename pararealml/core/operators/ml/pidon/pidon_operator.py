from typing import Sequence, Optional, Dict, Iterable, Tuple, NamedTuple, \
    List, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_condition import \
    VectorizedInitialConditionFunction
from pararealml.core.initial_value_problem import InitialValueProblem, \
    TemporalDomainInterval
from pararealml.core.operator import Operator, discretize_time_domain
from pararealml.core.operators.ml.pidon.collocation_point_sampler import \
    CollocationPointSampler
from pararealml.core.operators.ml.pidon.data_set import DataSet
from pararealml.core.operators.ml.pidon.loss import Loss
from pararealml.core.operators.ml.pidon.pi_deeponet import PIDeepONet
from pararealml.core.solution import Solution


class DataArgs(NamedTuple):
    """
    A container class for arguments pertaining to the generation and traversal
    of PIDON data sets.
    """
    y_0_functions: Iterable[VectorizedInitialConditionFunction]
    n_domain_points: int
    n_batches: int
    n_boundary_points: int = 0
    n_ic_repeats: int = 1


class ModelArgs(NamedTuple):
    """
    A container class for arguments pertaining to the architecture of a PIDON
    model.
    """
    latent_output_size: int
    branch_hidden_layer_sizes: Optional[List[int]] = None
    trunk_hidden_layer_sizes: Optional[List[int]] = None
    branch_initialization: str = 'glorot_uniform'
    trunk_initialization: str = 'glorot_uniform'
    branch_activation: Optional[str] = 'tanh'
    trunk_activation: Optional[str] = 'tanh'


class OptimizationArgs(NamedTuple):
    """
    A container class for arguments pertaining to the training of a PIDON
    model.
    """
    optimizer: Union[str, Dict[str, Any], Optimizer]
    epochs: int
    diff_eq_loss_weight: float = 1.
    ic_loss_weight: float = 1.
    bc_loss_weight: float = 1.
    verbose: bool = True


class SecondaryOptimizationArgs(NamedTuple):
    """
    A container class for arguments pertaining to the training of a PIDON
    model using a second order optimization method to fine tune the model
    parameters.
    """
    max_iterations: int = 50
    gradient_tol: float = 1e-8
    diff_eq_loss_weight: float = 1.
    ic_loss_weight: float = 1.
    bc_loss_weight: float = 1.
    verbose: bool = True


class PIDONOperator(Operator):
    """
    A physics-informed DeepONet based unsupervised machine learning operator
    for solving initial value problems.
    """

    def __init__(
            self,
            sampler: CollocationPointSampler,
            d_t: float,
            vertex_oriented: bool,
            auto_regression_mode: bool = False):
        """
        :param sampler: the collocation point sampler to use to generate the
            data to train and test models
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        :param auto_regression_mode: whether the operator is to function in
            auto-regression mode using its prediction at the previous time
            step as the initial conditions for its prediction at the next time
            step
        """
        if d_t <= 0.:
            raise ValueError('time step size must be greater than 0')

        self._sampler = sampler
        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._auto_regression_mode = auto_regression_mode

        self._model: Optional[PIDeepONet] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

    @property
    def auto_regression_mode(self) -> bool:
        """
        Whether the operator functions in auto-regression mode.
        """
        return self._auto_regression_mode

    @property
    def model(self) -> Optional[PIDeepONet]:
        """
        The physics-informed DeepONet model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[PIDeepONet]):
        self._model = model

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True)
            x_tensor = tf.convert_to_tensor(x, tf.float32)
            u = ivp.initial_condition.y_0(x).reshape((1, -1))
            u_tensor = tf.tile(
                tf.convert_to_tensor(u, tf.float32),
                (x.shape[0], 1))
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, tf.float32)

        t_tensor = tf.constant(
            self._d_t if self._auto_regression_mode else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1))

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_tensor = self._model.call((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regression_mode:
                    u_tensor = tf.tile(
                        tf.reshape(y_tensor, (1, -1)),
                        (x_tensor.shape[0], 1)) if diff_eq.x_dimension \
                        else tf.reshape(y_tensor, u_tensor.shape)
                else:
                    t_tensor = tf.constant(
                        t[i + 1],
                        dtype=tf.float32,
                        shape=(u_tensor.shape[0], 1))

        return Solution(
            ivp,
            t,
            y,
            vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)

    def train(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            *,
            training_data_args: DataArgs,
            model_args: ModelArgs,
            optimization_args: OptimizationArgs,
            test_data_args: Optional[DataArgs] = None,
            secondary_optimization_args: Optional[SecondaryOptimizationArgs] =
            None
    ) -> Tuple[Sequence[Loss], Sequence[Loss]]:
        """
        Trains a physics-informed DeepONet model on the provided constrained
        problem, time interval, and initial condition functions using the model
        and training arguments. It also saves the trained model to use as the
        predictor for solving IVPs.

        :param cp: the constrained problem to train the operator on
        :param t_interval: the time interval to train the operator on
        :param training_data_args: the training data generation and batch size
            arguments
        :param model_args: the physics-informed DeepONet model arguments
        :param optimization_args: the physics-informed DeepONet model
            optimization arguments
        :param test_data_args: the test data generation and batch size
            arguments
        :param secondary_optimization_args: the physics-informed DeepONet model
            optimization arguments for fine tuning the model parameters using
            a (quasi) second order optimization method
        :return: the training loss history and the test loss history
        """
        if self._auto_regression_mode and t_interval != (0., self._d_t):
            raise ValueError(
                'in auto-regression mode, the training time interval '
                f'{t_interval} must range from 0 to the time step size of '
                f'the operator ({self._d_t})')

        training_data_set = DataSet(
            cp,
            t_interval,
            point_sampler=self._sampler,
            y_0_functions=training_data_args.y_0_functions,
            n_domain_points=training_data_args.n_domain_points,
            n_boundary_points=training_data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented)
        training_data = training_data_set.get_iterator(
            training_data_args.n_batches, training_data_args.n_ic_repeats)

        if test_data_args:
            test_data_set = DataSet(
                cp,
                t_interval,
                point_sampler=self._sampler,
                y_0_functions=test_data_args.y_0_functions,
                n_domain_points=test_data_args.n_domain_points,
                n_boundary_points=test_data_args.n_boundary_points,
                vertex_oriented=self._vertex_oriented)
            test_data = test_data_set.get_iterator(
                test_data_args.n_batches,
                test_data_args.n_ic_repeats,
                shuffle=False)
        else:
            test_data = None

        model = PIDeepONet(
            cp, vertex_oriented=self._vertex_oriented, **model_args._asdict())
        loss_histories = model.fit(
            training_data=training_data,
            test_data=test_data,
            **optimization_args._asdict())
        if secondary_optimization_args:
            secondary_losses = model.fit_with_lbfgs(
                training_data=training_data,
                test_data=test_data,
                **secondary_optimization_args._asdict())
            loss_histories[0].append(secondary_losses[0])
            if secondary_losses[1] is not None:
                loss_histories[1].append(secondary_losses[1])

        self._model = model

        return loss_histories
