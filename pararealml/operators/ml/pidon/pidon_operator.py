from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import VectorizedInitialConditionFunction
from pararealml.initial_value_problem import (
    InitialValueProblem,
    TemporalDomainInterval,
)
from pararealml.operator import Operator, discretize_time_domain
from pararealml.operators.ml.pidon.collocation_point_sampler import (
    CollocationPointSampler,
)
from pararealml.operators.ml.pidon.data_set import DataSet
from pararealml.operators.ml.pidon.loss import Loss
from pararealml.operators.ml.pidon.pi_deeponet import PIDeepONet
from pararealml.solution import Solution


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
        auto_regression_mode: bool = False,
    ):
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
        super(PIDONOperator, self).__init__(d_t, vertex_oriented)

        self._sampler = sampler
        self._auto_regression_mode = auto_regression_mode

        self._model: Optional[PIDeepONet] = None

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
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            x_tensor = tf.convert_to_tensor(x, tf.float32)
            u = ivp.initial_condition.y_0(x).reshape((1, -1))
            u_tensor = tf.tile(
                tf.convert_to_tensor(u, tf.float32), (x.shape[0], 1)
            )
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, tf.float32)

        t_tensor = tf.constant(
            self._d_t if self._auto_regression_mode else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1),
        )

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_i_tensor = self._infer((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_i_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regression_mode:
                    u_tensor = (
                        tf.tile(
                            tf.reshape(y_i_tensor, (1, -1)),
                            (x_tensor.shape[0], 1),
                        )
                        if diff_eq.x_dimension
                        else tf.reshape(y_i_tensor, u_tensor.shape)
                    )
                else:
                    t_tensor = tf.constant(
                        t[i + 1],
                        dtype=tf.float32,
                        shape=(u_tensor.shape[0], 1),
                    )

        return Solution(
            ivp, t, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t
        )

    def train(
        self,
        cp: ConstrainedProblem,
        t_interval: TemporalDomainInterval,
        *,
        training_data_args: DataArgs,
        optimization_args: OptimizationArgs,
        model_args: Optional[ModelArgs] = None,
        test_data_args: Optional[DataArgs] = None,
        secondary_optimization_args: Optional[
            SecondaryOptimizationArgs
        ] = None,
    ) -> TrainingResults:
        """
        Trains a physics-informed DeepONet model on the provided constrained
        problem, time interval, and initial condition functions using the model
        and training arguments. It also saves the trained model to use as the
        predictor for solving IVPs.

        :param cp: the constrained problem to train the operator on
        :param t_interval: the time interval to train the operator on
        :param training_data_args: the training data generation and batch size
            arguments
        :param optimization_args: the physics-informed DeepONet model
            optimization arguments
        :param model_args: the physics-informed DeepONet model arguments; if
            the operator already has a model, it can be None
        :param test_data_args: the test data generation and batch size
            arguments
        :param secondary_optimization_args: the physics-informed DeepONet model
            optimization arguments for fine tuning the model parameters using
            a (quasi) second order optimization method
        :return: the training results
        """
        if model_args is None and self._model is None:
            raise ValueError(
                "the model arguments cannot be None if the operator's model "
                "is None"
            )

        if self._auto_regression_mode:
            if t_interval != (0.0, self._d_t):
                raise ValueError(
                    "in auto-regression mode, the training time interval "
                    f"{t_interval} must range from 0 to the time step size of "
                    f"the operator ({self._d_t})"
                )

            diff_eq = cp.differential_equation
            t_symbol = diff_eq.symbols.t
            eq_sys = diff_eq.symbolic_equation_system
            if any([t_symbol in rhs.free_symbols for rhs in eq_sys.rhs]):
                raise ValueError(
                    "auto-regression mode is not compatible with differential "
                    "equations whose right-hand sides contain any t terms"
                )

            if (
                diff_eq.x_dimension
                and not cp.are_all_boundary_conditions_static
            ):
                raise ValueError(
                    "auto-regression mode is not compatible with dynamic "
                    "boundary conditions"
                )

        training_data_set = DataSet(
            cp,
            t_interval,
            point_sampler=self._sampler,
            y_0_functions=training_data_args.y_0_functions,
            n_domain_points=training_data_args.n_domain_points,
            n_boundary_points=training_data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented,
        )
        training_data = training_data_set.get_iterator(
            training_data_args.n_batches,
            n_ic_repeats=training_data_args.n_ic_repeats,
            shuffle=training_data_args.shuffle,
        )

        if test_data_args:
            test_data_set = DataSet(
                cp,
                t_interval,
                point_sampler=self._sampler,
                y_0_functions=test_data_args.y_0_functions,
                n_domain_points=test_data_args.n_domain_points,
                n_boundary_points=test_data_args.n_boundary_points,
                vertex_oriented=self._vertex_oriented,
            )
            test_data = test_data_set.get_iterator(
                test_data_args.n_batches,
                n_ic_repeats=test_data_args.n_ic_repeats,
                shuffle=test_data_args.shuffle,
            )
        else:
            test_data = None

        model = (
            self._model
            if model_args is None
            else PIDeepONet(
                cp=cp,
                vertex_oriented=self._vertex_oriented,
                **model_args._asdict(),
            )
        )

        training_loss_history, test_loss_history = model.train(
            training_data=training_data,
            test_data=test_data,
            **optimization_args._asdict(),
        )

        if secondary_optimization_args:
            model.train_with_lbfgs(
                training_data=training_data,
                **secondary_optimization_args._asdict(),
            )

        final_training_loss = model.evaluate(training_data)
        final_test_loss = model.evaluate(test_data) if test_data else None

        self._model = model

        return TrainingResults(
            training_loss_history,
            test_loss_history,
            final_training_loss,
            final_test_loss,
        )

    @tf.function
    def _infer(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
    ) -> tf.Tensor:
        """
        Propagates the inputs through the physics-informed DeepONet.

        :param inputs: the model inputs
        :return: the model outputs
        """
        return self.model.__call__(inputs)


class DataArgs(NamedTuple):
    """
    Arguments pertaining to the generation and traversal of PIDON data sets.
    """

    y_0_functions: Iterable[VectorizedInitialConditionFunction]
    n_domain_points: int
    n_batches: int
    n_boundary_points: int = 0
    n_ic_repeats: int = 1
    shuffle: bool = True


class ModelArgs(NamedTuple):
    """
    Arguments pertaining to the architecture of a PIDON model.
    """

    branch_net: tf.keras.Model
    trunk_net: tf.keras.Model
    combiner_net: tf.keras.Model
    diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0
    ic_loss_weight: Union[float, Sequence[float]] = 1.0
    bc_loss_weight: Union[float, Sequence[float]] = 1.0


class OptimizationArgs(NamedTuple):
    """
    Arguments pertaining to the training of a PIDON model.
    """

    optimizer: Union[str, Dict[str, Any], tf.optimizers.Optimizer]
    epochs: int
    restore_best_weights: bool = True


class SecondaryOptimizationArgs(NamedTuple):
    """
    Arguments pertaining to the training of a PIDON model using a second order
    optimization method to fine tune the model parameters.
    """

    max_iterations: int = 50
    max_line_search_iterations: int = 50
    parallel_iterations: int = 1
    num_correction_pairs: int = 10
    gradient_tol: float = 1e-8


class TrainingResults(NamedTuple):
    """
    The results of the training of the PIDON operator.
    """

    training_loss_history: List[Loss]
    test_loss_history: Optional[List[Loss]]
    final_training_loss: Loss
    final_test_loss: Optional[Loss]
