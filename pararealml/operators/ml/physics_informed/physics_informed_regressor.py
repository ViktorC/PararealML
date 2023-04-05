from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import LHS
from pararealml.operators.ml.physics_informed.auto_differentiator import (
    AutoDifferentiator,
)
from pararealml.operators.ml.physics_informed.physics_informed_ml_symbol_mapper import (  # noqa: 501
    PhysicsInformedMLSymbolMapArg,
    PhysicsInformedMLSymbolMapFunction,
    PhysicsInformedMLSymbolMapper,
)


class PhysicsInformedRegressor(tf.keras.Model):
    """
    A physics-informed regression model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        cp: ConstrainedProblem,
        diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0,
        ic_loss_weight: Union[float, Sequence[float]] = 1.0,
        bc_loss_weight: Union[float, Sequence[float]] = 1.0,
        vertex_oriented: bool = False,
    ):
        """
        :param model: the base regression model
        :param cp: the constrained problem to build a physics-informed neural
            network around
        :param diff_eq_loss_weight: the weight of the differential equation
            violation term of the physics-informed loss
        :param ic_loss_weight: the weight of the initial condition violation
            term of the physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition violation
            terms of the total physics-informed loss; if the constrained
            is an ODE, it is ignored
        :param vertex_oriented: whether the initial condition collocation
            points are the vertices or the cell centers of the mesh
        """
        diff_eq = cp.differential_equation
        x_dim = diff_eq.x_dimension
        y_dim = diff_eq.y_dimension

        inputs = tf.keras.layers.Input(
            shape=(np.prod(cp.y_shape(vertex_oriented)) + x_dim + 1,)
        )
        base_model_output_shape = tf.keras.Model(
            inputs=inputs, outputs=model.call(inputs)
        ).output.shape
        if base_model_output_shape != (None, y_dim):
            raise ValueError(
                "base regression model output shape "
                f"{base_model_output_shape} must be {(None, y_dim)}"
            )

        diff_eq_loss_weights = (
            (diff_eq_loss_weight,) * y_dim
            if isinstance(diff_eq_loss_weight, float)
            else tuple(diff_eq_loss_weight)
        )
        ic_loss_weights = (
            (ic_loss_weight,) * y_dim
            if isinstance(ic_loss_weight, float)
            else tuple(ic_loss_weight)
        )
        bc_loss_weights = (
            (bc_loss_weight,) * y_dim
            if isinstance(bc_loss_weight, float)
            else tuple(bc_loss_weight)
        )

        if (
            len(diff_eq_loss_weights) != y_dim
            or len(ic_loss_weights) != y_dim
            or len(bc_loss_weights) != y_dim
        ):
            raise ValueError(
                f"length of all loss weights must match y dimension ({y_dim})"
            )

        super(PhysicsInformedRegressor, self).__init__()

        self._model = model
        self._cp = cp
        self._diff_eq_loss_weights = diff_eq_loss_weights
        self._ic_loss_weights = ic_loss_weights
        self._bc_loss_weights = bc_loss_weights

        self._symbol_mapper = PhysicsInformedMLSymbolMapper(cp)
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()

        self._primary_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._diff_eq_loss_trackers = [
            tf.keras.metrics.Mean(name=f"diff_eq_loss_{i}")
            for i in range(y_dim)
        ]
        self._ic_loss_trackers = [
            tf.keras.metrics.Mean(name=f"ic_loss_{i}") for i in range(y_dim)
        ]
        self._dirichlet_bc_loss_trackers = [
            tf.keras.metrics.Mean(name=f"dirichlet_bc_loss_{i}")
            for i in range(y_dim)
        ]
        self._neumann_bc_loss_trackers = [
            tf.keras.metrics.Mean(name=f"neumann_bc_loss_{i}")
            for i in range(y_dim)
        ]
        self._model_loss_tracker = tf.keras.metrics.Mean(name="model_loss")

    @property
    def model(self) -> tf.keras.Model:
        """
        The base regression model.
        """
        return self._model

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the model is built around.
        """
        return self._cp

    @property
    def differential_equation_loss_weights(self) -> Sequence[float]:
        """
        The weights of the differential equation violation term of the
        physics-informed loss.
        """
        return self._diff_eq_loss_weights

    @property
    def initial_condition_loss_weights(self) -> Sequence[float]:
        """
        The weights of the initial condition violation term of the
        physics-informed loss.
        """
        return self._ic_loss_weights

    @property
    def boundary_condition_loss_weights(self) -> Sequence[float]:
        """
        The weights of the boundary condition violation terms of the
        physics-informed loss.
        """
        return self._bc_loss_weights

    @property
    def metrics(self) -> Sequence[tf.keras.metrics.Metric]:
        """
        The metrics tracked during model fitting and evaluation.
        """
        return (
            [self._primary_loss_tracker]
            + self._diff_eq_loss_trackers
            + self._ic_loss_trackers
            + self._dirichlet_bc_loss_trackers
            + self._neumann_bc_loss_trackers
            + [self._model_loss_tracker]
        )

    def call(
        self,
        inputs: Union[
            tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
        ],
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if isinstance(inputs, tuple):
            u = inputs[0]
            t = inputs[1]
            x = inputs[2]
            input_tensor = tf.concat((u, t) if x is None else inputs, axis=1)
        else:
            input_tensor = inputs
        return self._model(input_tensor, training=training, mask=mask)

    def train_step(
        self, data: Dict[str, Dict[str, tf.Tensor]]
    ) -> Dict[str, np.ndarray]:
        with tf.GradientTape() as tape:
            loss = self._compute_batch_loss(data, training=True)
        self.optimizer.apply_gradients(
            zip(
                tape.gradient(loss, self.trainable_variables),
                self.trainable_variables,
            )
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(
        self, data: Dict[str, Dict[str, tf.Tensor]]
    ) -> Dict[str, np.ndarray]:
        self._compute_batch_loss(data, training=False)
        return {metric.name: metric.result() for metric in self.metrics}

    def _create_diff_eq_lhs_functions(
        self,
    ) -> Sequence[PhysicsInformedMLSymbolMapFunction]:
        """
        Creates a sequence of symbol map functions representing the left-hand
        side of the differential equation system.
        """
        diff_eq = self._cp.differential_equation

        lhs_functions = []
        for y_ind, lhs_type in enumerate(
            diff_eq.symbolic_equation_system.lhs_types
        ):
            if lhs_type == LHS.D_Y_OVER_D_T:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.auto_diff.batch_gradient(
                        arg.t, arg.y_hat[:, _y_ind : _y_ind + 1], 0
                    )
                )

            elif lhs_type == LHS.Y:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.y_hat[:, _y_ind : _y_ind + 1]
                )

            elif lhs_type == LHS.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.auto_diff.batch_laplacian(
                        arg.x,
                        arg.y_hat[:, _y_ind : _y_ind + 1],
                        self._cp.mesh.coordinate_system_type,
                    )
                )

            else:
                raise ValueError(
                    f"unsupported left-hand side type ({lhs_type.name})"
                )

        return lhs_functions

    def _compute_batch_loss(
        self,
        batch: Dict[str, Dict[str, tf.Tensor]],
        training: bool,
    ) -> tf.Tensor:
        """
        Computes and returns the physics-informed loss over the batch
        consisting of the mean squared differential equation error, the mean
        squared initial condition error, in the case of PDEs, the mean
        squared Dirichlet and Neumann boundary condition errors, and if
        applicable, the regularization error.

        :param batch: the batch to compute the losses over
        :param training: whether to perform the loss computation in training
            mode
        :return: the loss over the batch
        """
        diff_eq = self._cp.differential_equation

        diff_eq_loss = self._compute_differential_equation_loss(
            batch["domain"], training
        )
        weighted_total_loss = tf.multiply(
            tf.constant(self._diff_eq_loss_weights), diff_eq_loss
        )

        ic_loss = self._compute_initial_condition_loss(
            batch["initial"], training
        )
        weighted_total_loss += tf.multiply(
            tf.constant(self._ic_loss_weights), ic_loss
        )

        for i in range(diff_eq.y_dimension):
            self._diff_eq_loss_trackers[i].update_state(diff_eq_loss[i])
            self._ic_loss_trackers[i].update_state(ic_loss[i])

        if diff_eq.x_dimension:
            (
                dirichlet_bc_loss,
                neumann_bc_loss,
            ) = self._compute_boundary_condition_loss(
                batch["boundary"], training
            )
            weighted_total_loss += tf.multiply(
                tf.constant(self._bc_loss_weights),
                dirichlet_bc_loss + neumann_bc_loss,
            )

            for i in range(diff_eq.y_dimension):
                self._dirichlet_bc_loss_trackers[i].update_state(
                    dirichlet_bc_loss[i]
                )
                self._neumann_bc_loss_trackers[i].update_state(
                    neumann_bc_loss[i]
                )

        if self.losses:
            model_loss = tf.reshape(tf.add_n(self.losses), (1,))
            weighted_total_loss += model_loss

            self._model_loss_tracker.update_state(model_loss)

        loss = tf.math.reduce_sum(weighted_total_loss)
        self._primary_loss_tracker.update_state(loss)
        return loss

    def _compute_differential_equation_loss(
        self, domain_batch: Dict[str, tf.Tensor], training: bool
    ) -> tf.Tensor:
        """
        Computes and returns the mean squared differential equation error.

        :param domain_batch: the domain data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared differential equation error
        """
        u = domain_batch["u"]
        t = domain_batch["t"]
        x = domain_batch.get("x", None)

        with AutoDifferentiator(persistent=True) as auto_diff:
            auto_diff.watch(t)
            if x is not None:
                auto_diff.watch(x)

            y_hat = self.__call__((u, t, x), training=training)

            symbol_map_arg = PhysicsInformedMLSymbolMapArg(
                auto_diff, t, x, y_hat
            )
            rhs = self._symbol_mapper.map(symbol_map_arg)

            diff_eq_residual = tf.concat(
                [
                    self._diff_eq_lhs_functions[i](symbol_map_arg) - rhs[i]
                    for i in range(len(rhs))
                ],
                axis=1,
            )

        squared_diff_eq_error = tf.square(diff_eq_residual)
        return tf.reduce_mean(squared_diff_eq_error, axis=0)

    def _compute_initial_condition_loss(
        self, initial_batch: Dict[str, tf.Tensor], training: bool
    ) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param initial_batch: the initial condition data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared initial condition error
        """
        y_hat = self.__call__(
            (
                initial_batch["u"],
                initial_batch["t"],
                initial_batch.get("x", None),
            ),
            training=training,
        )
        squared_ic_error = tf.square(y_hat - initial_batch["y"])
        return tf.reduce_mean(squared_ic_error, axis=0)

    def _compute_boundary_condition_loss(
        self, boundary_batch: Dict[str, tf.Tensor], training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the mean squared Dirichlet boundary condition
        error and the mean squared Neumann boundary condition error.

        :param boundary_batch: the boundary data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared Dirichlet and Neumann boundary condition
            errors
        """
        u = boundary_batch["u"]
        t = boundary_batch["t"]
        x = boundary_batch["x"]
        y = boundary_batch["y"]
        d_y_over_d_n = boundary_batch["d_y_over_d_n"]
        axes = boundary_batch["axes"]

        with AutoDifferentiator() as auto_diff:
            auto_diff.watch(x)
            y_hat = self.__call__((u, t, x), training=training)

        d_y_over_d_n_hat = auto_diff.batch_gradient(x, y_hat, axes)

        dirichlet_bc_error = y_hat - y
        dirichlet_bc_error = tf.where(
            tf.math.is_nan(y), tf.zeros_like(y), dirichlet_bc_error
        )
        squared_dirichlet_bc_error = tf.square(dirichlet_bc_error)
        mean_squared_dirichlet_bc_error = tf.reduce_mean(
            squared_dirichlet_bc_error, axis=0
        )

        neumann_bc_error = d_y_over_d_n_hat - d_y_over_d_n
        neumann_bc_error = tf.where(
            tf.math.is_nan(d_y_over_d_n),
            tf.zeros_like(d_y_over_d_n),
            neumann_bc_error,
        )
        squared_neumann_bc_error = tf.square(neumann_bc_error)
        mean_squared_neumann_bc_error = tf.reduce_mean(
            squared_neumann_bc_error, axis=0
        )

        return mean_squared_dirichlet_bc_error, mean_squared_neumann_bc_error
