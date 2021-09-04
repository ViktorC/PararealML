from typing import Callable, Sequence, Optional, Dict, Iterable, Tuple, \
    NamedTuple, List, Union, Any

import numpy as np
import tensorflow as tf

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_value_problem import InitialValueProblem, \
    TemporalDomainInterval
from pararealml.core.operator import Operator, discretize_time_domain
from pararealml.core.operators.pidon.collocation_point_sampler import \
    CollocationPointSampler
from pararealml.core.operators.pidon.data_set import DataSet
from pararealml.core.operators.pidon.loss import Loss
from pararealml.core.operators.pidon.pi_deeponet import PIDeepONet
from pararealml.core.solution import Solution


class DataArgs(NamedTuple):
    """
    A container class for arguments pertaining to the generation and traversal
    of PIDON data sets.
    """
    y_0_functions: Iterable[
        Callable[[Optional[Sequence[float]]], Sequence[float]]
    ]
    n_domain_points: int
    domain_batch_size: int
    n_boundary_points: int = 0
    boundary_batch_size: int = 0


class ModelArgs(NamedTuple):
    """
    A container class for arguments pertaining to the architecture of a PIDON
    model.
    """
    latent_output_size: int
    branch_hidden_layer_sizes: Optional[List[int]] = None
    trunk_hidden_layer_sizes: Optional[List[int]] = None
    branch_initialization: Optional[str] = None
    trunk_initialization: Optional[str] = None
    branch_activation: Optional[str] = 'tanh'
    trunk_activation: Optional[str] = 'tanh'


class OptimizationArgs(NamedTuple):
    """
    A container class for arguments pertaining to the training of a PIDON
    model.
    """
    optimizer: Union[str, Dict[str, Any]]
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
            vertex_oriented: bool):
        """
        :param sampler: the collocation point sampler to use to generate the
            data to train and test models
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        """
        if d_t <= 0.:
            raise ValueError

        self._sampler = sampler
        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._model: Optional[PIDeepONet] = None

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented

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
            parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        time_points = discretize_time_domain(ivp.t_interval, self._d_t)

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True)
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            u = cp.mesh.evaluate(
                [ivp.initial_condition.y_0],
                False,
                True)
            u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
            u_tensor = tf.tile(u_tensor, (x.shape[0], 1))
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(time_points) - 1,) + y_shape)

        for i, t_i in enumerate(time_points[1:]):
            t_tensor = tf.tile(
                tf.convert_to_tensor([[t_i]], dtype=tf.float32),
                (u_tensor.shape[0], 1))
            y_tensor = self._model.predict(u_tensor, t_tensor, x_tensor)
            y[i, ...] = y_tensor.numpy().reshape(y_shape)

        return Solution(
            ivp,
            time_points[1:],
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
        training_data_set = DataSet(
            cp,
            t_interval,
            point_sampler=self._sampler,
            y_0_functions=training_data_args.y_0_functions,
            n_domain_points=training_data_args.n_domain_points,
            n_boundary_points=training_data_args.n_boundary_points)
        training_data = training_data_set.get_iterator(
            domain_batch_size=training_data_args.domain_batch_size,
            boundary_batch_size=training_data_args.boundary_batch_size)

        if test_data_args:
            test_data_set = DataSet(
                cp,
                t_interval,
                point_sampler=self._sampler,
                y_0_functions=test_data_args.y_0_functions,
                n_domain_points=test_data_args.n_domain_points,
                n_boundary_points=test_data_args.n_boundary_points)
            test_data = test_data_set.get_iterator(
                domain_batch_size=test_data_args.domain_batch_size,
                boundary_batch_size=test_data_args.boundary_batch_size,
                shuffle=False)
        else:
            test_data = None

        model = PIDeepONet(cp, **model_args._asdict())
        model.init()

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
