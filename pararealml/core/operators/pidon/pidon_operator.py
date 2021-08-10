from typing import Callable, Sequence, Optional, Dict, Iterable, Tuple, \
    NamedTuple

import numpy as np
import tensorflow as tf

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_value_problem import InitialValueProblem, \
    TemporalDomainInterval
from pararealml.core.operator import Operator, discretise_time_domain
from pararealml.core.operators.pidon.collocation_point_sampler import \
    CollocationPointSampler
from pararealml.core.operators.pidon.data_set import DataSet
from pararealml.core.operators.pidon.pi_deeponet import PIDeepONet
from pararealml.core.solution import Solution


class LossArrays(NamedTuple):
    """
    A collection of the various losses of a physics-informed DeepONet in
    array form.
    """
    total_weighted_loss: np.ndarray
    diff_eq_loss: np.ndarray
    ic_loss: np.ndarray
    dirichlet_bc_loss: np.ndarray
    neumann_bc_loss: np.ndarray


class PIDONOperator(Operator):
    """
    A physics informed DeepONet based unsupervised machine learning operator
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

        time_points = discretise_time_domain(ivp.t_interval, self._d_t)

        if diff_eq.x_dimension:
            x = cp.mesh.all_x(self._vertex_oriented)
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            u = cp.mesh.evaluate_fields(
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
            cp,
            time_points[1:],
            y,
            vertex_oriented=self._vertex_oriented,
            d_t=self._d_t)

    def train(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            training_y_0_functions: Iterable[
                Callable[[Optional[Sequence[float]]], Sequence[float]]
            ],
            model_arguments: Dict,
            training_arguments: Dict,
            n_training_domain_points: int,
            training_domain_batch_size: int,
            n_training_boundary_points: int = 0,
            training_boundary_batch_size: int = 0,
            n_test_domain_points: int = 0,
            test_domain_batch_size: int = 0,
            n_test_boundary_points: int = 0,
            test_boundary_batch_size: int = 0,
            test_y_0_functions: Optional[Iterable[
                Callable[[Optional[Sequence[float]]], Sequence[float]]
            ]] = None
    ) -> Tuple[Sequence[LossArrays], Sequence[LossArrays]]:
        """
        Trains a physics-informed DeepONet model on the provided constrained
        problem, time interval, and initial condition functions using the model
        and training arguments. It also saves the trained model to use as the
        predictor for solving IVPs.

        :param cp: the constrained problem to train the operator on
        :param t_interval: the time interval to train the operator on
        :param training_y_0_functions: the set of initial condition functions
            to train the operator on
        :param model_arguments: the physics-informed DeepONet model arguments
        :param training_arguments: the physics informed DeepONet model training
            arguments
        :param n_training_domain_points: the number of domain points to
            generate for the training of the physics-informed DeepONet model
        :param training_domain_batch_size: the training domain data batch size
        :param n_training_boundary_points: the number of boundary points to
            generate for the training of the physics-informed DeepONet model
        :param training_boundary_batch_size: the training boundary data batch
            size
        :param n_test_domain_points: the number of domain points to
            generate for the testing of the physics-informed DeepONet model
        :param test_domain_batch_size: the test domain data batch size
        :param n_test_boundary_points: the number of boundary points to
            generate for the testing of the physics-informed DeepONet model
        :param test_boundary_batch_size: the test boundary data batch size
        :param test_y_0_functions: the set of initial condition functions to
            test the operator on
        :return: the training loss history and the test loss history
        """
        model = PIDeepONet(cp, **model_arguments)
        model.init()

        training_data_set = DataSet(
            cp,
            t_interval,
            training_y_0_functions,
            self._sampler,
            n_training_domain_points,
            n_training_boundary_points)
        training_data = training_data_set.get_iterator(
            training_domain_batch_size, training_boundary_batch_size)

        if n_test_domain_points > 0:
            test_data_set = DataSet(
                cp,
                t_interval,
                training_y_0_functions if test_y_0_functions is None
                else test_y_0_functions,
                self._sampler,
                n_test_domain_points,
                n_test_boundary_points)
            test_data = test_data_set.get_iterator(
                test_domain_batch_size,
                test_boundary_batch_size,
                shuffle=False)
        else:
            test_data = None

        loss_tensor_histories = model.train(
            training_data=training_data,
            test_data=test_data,
            **training_arguments)

        self._model = model

        loss_array_histories = [
            [
                LossArrays(
                    loss_tensors.total_weighted_loss.numpy(),
                    loss_tensors.diff_eq_loss.numpy(),
                    loss_tensors.ic_loss.numpy(),
                    loss_tensors.dirichlet_bc_loss.numpy(),
                    loss_tensors.neumann_bc_loss.numpy())
                for loss_tensors in loss_tensor_history
            ] for loss_tensor_history in loss_tensor_histories
        ]
        return loss_array_histories[0], loss_array_histories[1]
