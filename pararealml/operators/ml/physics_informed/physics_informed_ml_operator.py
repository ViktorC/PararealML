from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
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
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    CollocationPointSampler,
)
from pararealml.operators.ml.physics_informed.dataset_generator import (
    DatasetGenerator,
)
from pararealml.operators.ml.physics_informed.physics_informed_regressor import (  # noqa: 501
    PhysicsInformedRegressor,
)
from pararealml.solution import Solution


class PhysicsInformedMLOperator(Operator):
    """
    A physics-informed machine learning operator for solving initial value
    problems.
    """

    def __init__(
        self,
        sampler: CollocationPointSampler,
        d_t: float,
        vertex_oriented: bool,
        auto_regressive: bool = False,
    ):
        """
        :param sampler: the collocation point sampler to use to generate the
            data to train and test models
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        :param auto_regressive: whether the operator is to function in
            auto-regressive mode using its prediction at the previous time
            step as the initial conditions for its prediction at the next time
            step
        """
        super(PhysicsInformedMLOperator, self).__init__(d_t, vertex_oriented)
        self._sampler = sampler
        self._auto_regressive = auto_regressive
        self._model: Optional[PhysicsInformedRegressor] = None

    @property
    def auto_regressive(self) -> bool:
        """
        Whether the operator functions in auto-regressive mode.
        """
        return self._auto_regressive

    @property
    def model(self) -> Optional[PhysicsInformedRegressor]:
        """
        The physics-informed regresion model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[PhysicsInformedRegressor]):
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
            self._d_t if self._auto_regressive else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1),
        )

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_i_tensor = self._infer((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_i_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regressive:
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
        training_data_args: DataArgs,
        optimization_args: OptimizationArgs,
        model_args: Optional[ModelArgs] = None,
        validation_data_args: Optional[DataArgs] = None,
        test_data_args: Optional[DataArgs] = None,
    ) -> Tuple[tf.keras.callbacks.History, Optional[Sequence[float]]]:
        """
        Trains a physics-informed regresion model on the provided constrained
        problem, time interval, and initial condition functions using the model
        and training arguments. It also saves the trained model to use as the
        predictor for solving IVPs.

        :param cp: the constrained problem to train the operator on
        :param t_interval: the time interval to train the operator on
        :param training_data_args: the training data generation arguments
        :param optimization_args: the physics-informed regresion model
            optimization arguments
        :param model_args: the physics-informed regresion model arguments; if
            the operator already has a model, it can be None
        :param validation_data_args: the validation data generation arguments
        :param test_data_args: the test data generation arguments
        :return: the training history and optionally the test loss
        """
        if model_args is None and self._model is None:
            raise ValueError(
                "the model arguments cannot be None if the operator's model "
                "is None"
            )

        if self._auto_regressive:
            if t_interval != (0.0, self._d_t):
                raise ValueError(
                    "in auto-regressive mode, the training time interval "
                    f"{t_interval} must range from 0 to the time step size of "
                    f"the operator ({self._d_t})"
                )

            diff_eq = cp.differential_equation
            t_symbol = diff_eq.symbols.t
            eq_sys = diff_eq.symbolic_equation_system
            if any([t_symbol in rhs.free_symbols for rhs in eq_sys.rhs]):
                raise ValueError(
                    "auto-regressive mode is not compatible with differential "
                    "equations whose right-hand sides contain any t terms"
                )

            if (
                diff_eq.x_dimension
                and not cp.are_all_boundary_conditions_static
            ):
                raise ValueError(
                    "auto-regressive mode is not compatible with dynamic "
                    "boundary conditions"
                )

        training_dataset = self._create_dataset(
            cp, t_interval, training_data_args
        )
        validation_dataset = self._create_dataset(
            cp, t_interval, validation_data_args
        )
        test_dataset = self._create_dataset(cp, t_interval, test_data_args)

        model = (
            self._model
            if model_args is None
            else PhysicsInformedRegressor(
                cp=cp,
                model=model_args.model,
                diff_eq_loss_weight=model_args.diff_eq_loss_weight,
                ic_loss_weight=model_args.ic_loss_weight,
                bc_loss_weight=model_args.bc_loss_weight,
                vertex_oriented=self._vertex_oriented,
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.get(optimization_args.optimizer)
        )
        history = model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=optimization_args.epochs,
            callbacks=optimization_args.callbacks,
            verbose=optimization_args.verbose,
        )

        test_loss = (
            model.evaluate(test_dataset, verbose=optimization_args.verbose)
            if test_dataset
            else None
        )

        self._model = model

        return history, test_loss

    @tf.function
    def _infer(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
    ) -> tf.Tensor:
        """
        Propagates the inputs through the physics-informed regression model.

        :param inputs: the model inputs
        :return: the model outputs
        """
        return self.model.__call__(inputs)

    def _create_dataset(
        self,
        cp: ConstrainedProblem,
        t_interval: Tuple[float, float],
        data_args: Optional[DataArgs],
    ) -> Optional[tf.data.Dataset]:
        """
        Creates a Tensorflow dataset given the constrained problem, time
        domain and that data arguments.

        :param cp: the constrained problem
        :param t_interval: the time domain
        :param data_args: the data generation arguments
        :return: a Tensorflow dataset
        """
        if not data_args:
            return None

        dataset = DatasetGenerator(
            cp=cp,
            t_interval=t_interval,
            y_0_functions=data_args.y_0_functions,
            point_sampler=self._sampler,
            n_domain_points=data_args.n_domain_points,
            n_boundary_points=data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented,
        ).generate(
            n_batches=data_args.n_batches,
            n_ic_repeats=data_args.n_ic_repeats,
            shuffle=data_args.shuffle,
            n_parallel_map_calls=data_args.n_parallel_map_calls,
            deterministic_mapped_order=data_args.deterministic_mapped_order,
        )
        if data_args.cache:
            dataset = dataset.cache()
        return dataset.prefetch(data_args.prefetch_buffer_size)


class DataArgs(NamedTuple):
    """
    Arguments pertaining to the generation of physics-informed regresion
    datasets.
    """

    y_0_functions: Iterable[VectorizedInitialConditionFunction]
    n_domain_points: int
    n_batches: int
    n_boundary_points: int = 0
    n_ic_repeats: int = 1
    shuffle: bool = True
    prefetch_buffer_size: int = 1
    n_parallel_map_calls: int = 1
    deterministic_mapped_order: bool = True
    cache: bool = False
    cache_file_path: str = ""


class ModelArgs(NamedTuple):
    """
    Arguments pertaining to the architecture of the physics-informed regresion
    model.
    """

    model: tf.keras.Model
    diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0
    ic_loss_weight: Union[float, Sequence[float]] = 1.0
    bc_loss_weight: Union[float, Sequence[float]] = 1.0


class OptimizationArgs(NamedTuple):
    """
    Arguments pertaining to the training of the physics-informed regresion
    model.
    """

    optimizer: Union[str, Dict[str, Any], tf.optimizers.Optimizer]
    epochs: int
    callbacks: Sequence[tf.keras.callbacks.Callback] = ()
    verbose: Union[str, int] = "auto"
