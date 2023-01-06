import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import LHS
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.pidon.auto_differentiator import (
    AutoDifferentiator,
)
from pararealml.operators.ml.pidon.data_set import (
    BoundaryDataBatch,
    DataBatch,
    DataSetIterator,
    DomainDataBatch,
    InitialDataBatch,
)
from pararealml.operators.ml.pidon.loss import Loss
from pararealml.operators.ml.pidon.pidon_symbol_mapper import (
    PIDONSymbolMapArg,
    PIDONSymbolMapFunction,
    PIDONSymbolMapper,
)


class PIDeepONet(DeepONet):
    """
    A Physics-Informed DeepONet model.

    See: https://arxiv.org/abs/2103.10974
    """

    def __init__(
        self,
        branch_net: tf.keras.Model,
        trunk_net: tf.keras.Model,
        combiner_net: tf.keras.Model,
        cp: ConstrainedProblem,
        diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0,
        ic_loss_weight: Union[float, Sequence[float]] = 1.0,
        bc_loss_weight: Union[float, Sequence[float]] = 1.0,
        vertex_oriented: bool = False,
    ):
        """
        :param branch_net: the model's branch net that processes the initial
            condition sensor readings
        :param trunk_net: the model's trunk net that processes the domain
            coordinates
        :param combiner_net: the model's combiner net that combines the outputs
            of the branch and trunk nets
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

        branch_net_output_shape = branch_net.compute_output_shape(
            (None, np.prod(cp.y_shape(vertex_oriented)))
        )
        trunk_net_output_shape = trunk_net.compute_output_shape(
            (None, x_dim + 1)
        )
        if branch_net_output_shape != trunk_net_output_shape:
            raise ValueError(
                f"branch net output shape {branch_net_output_shape} and "
                f"trunk net output shape {trunk_net_output_shape} must match"
            )

        combiner_net_output_shape = combiner_net.compute_output_shape(
            (None,) + tuple(3 * np.array(branch_net_output_shape[1:]))
        )
        if combiner_net_output_shape != (None, y_dim):
            raise ValueError(
                f"combiner net output shape {combiner_net_output_shape} "
                f"must be {(None, y_dim)}"
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

        super(PIDeepONet, self).__init__(branch_net, trunk_net, combiner_net)

        self._cp = cp
        self._diff_eq_loss_weights = diff_eq_loss_weights
        self._ic_loss_weights = ic_loss_weights
        self._bc_loss_weights = bc_loss_weights

        self._symbol_mapper = PIDONSymbolMapper(cp)
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()
        self._logger = logging.getLogger(__name__)

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

    def train(
        self,
        epochs: int,
        optimizer: Union[str, Dict[str, Any], tf.optimizers.Optimizer],
        training_data: DataSetIterator,
        test_data: Optional[DataSetIterator] = None,
        restore_best_weights: bool = True,
    ) -> Tuple[List[Loss], Optional[List[Loss]]]:
        """
        Fits the model by minimising the physics-informed loss function over
        the provided training data set with respect to the parameters of the
        branch, trunk, and combiner networks. It also evaluates the loss over
        both the training data and the test data, if provided, for every epoch.

        :param epochs: the number of epochs over the training data
        :param optimizer: the optimizer to use to minimize the loss function
        :param training_data: an iterator over the training data set
        :param test_data: an iterator over the test data set
        :param restore_best_weights: whether to restore the model parameters
            to those with the lowest test loss; if no test data is provided,
            this parameter is ignored
        :return: the training and test loss histories
        """
        if epochs < 1:
            raise ValueError("number of epochs must be greater than 0")

        optimizer_instance = tf.keras.optimizers.get(optimizer)

        best_test_loss_sum = None
        best_weights = None
        best_epoch = 0

        training_loss_history = []
        test_loss_history = [] if test_data else None

        self._logger.info("Gradient Descent Optimization")
        for epoch in range(epochs):
            self._logger.info(f"Epoch: {epoch}")

            training_loss = self._compute_and_minimize_total_loss(
                training_data, optimizer=optimizer_instance
            )
            self._logger.info("Training MSE - %s", training_loss)
            training_loss_history.append(training_loss)

            if test_data:
                test_loss = self._compute_and_minimize_total_loss(
                    test_data, optimizer=None
                )
                self._logger.info("Test MSE -  %s", test_loss)
                test_loss_history.append(test_loss)

                if restore_best_weights:
                    test_loss_sum = tf.math.reduce_sum(
                        test_loss.weighted_total_loss
                    )
                    if (
                        best_test_loss_sum is None
                        or test_loss_sum <= best_test_loss_sum
                    ):
                        best_test_loss_sum = test_loss_sum
                        best_weights = self.get_trainable_parameters()
                        best_epoch = epoch

        if test_data and restore_best_weights:
            self.set_trainable_parameters(best_weights)
            self._logger.info("Best Epoch: %s", best_epoch)
            self._logger.info(
                "Training MSE - %s", training_loss_history[best_epoch]
            )
            self._logger.info("Test MSE - %s", test_loss_history[best_epoch])

        return training_loss_history, test_loss_history

    def train_with_lbfgs(
        self,
        training_data: DataSetIterator,
        max_iterations: int,
        max_line_search_iterations: int,
        parallel_iterations: int,
        num_correction_pairs: int,
        gradient_tol: float,
    ):
        """
        Fits the model by minimising the physics-informed loss function over
        the provided training data set with respect to the parameters of the
        branch, trunk, and combiner networks using the L-BFGS optimization
        method.

        :param training_data: the data set providing the full training batch
        :param max_iterations: the maximum number of iterations to perform the
            optimization for
        :param max_line_search_iterations: the maximum number of line search
            iterations
        :param parallel_iterations: the number of iterations allowed to run in
            parallel
        :param num_correction_pairs: the maximum number of correction pairs
            (position update and gradient update) to keep as implicit
            approximation of the Hessian
        :param gradient_tol: the threshold on the gradient vector; if the
            largest element of the absolute value of the gradient vector is
            less than this threshold, the optimization is stopped
        """
        full_training_data_batch = training_data.get_full_batch()

        @tf.function
        def value_and_gradients_function(
            parameters: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            self.set_trainable_parameters(parameters)
            with AutoDifferentiator() as auto_diff:
                loss = self._compute_batch_loss(full_training_data_batch, True)
                value = tf.reduce_sum(loss.weighted_total_loss, keepdims=True)

            gradients = auto_diff.gradient(value, self.trainable_variables)
            flattened_gradients = tf.concat(
                [tf.reshape(gradient, (1, -1)) for gradient in gradients],
                axis=1,
            )
            return value, flattened_gradients

        self._logger.info("L-BFGS Optimization")
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_gradients_function,
            initial_position=self.get_trainable_parameters(),
            max_iterations=max_iterations,
            max_line_search_iterations=max_line_search_iterations,
            parallel_iterations=parallel_iterations,
            num_correction_pairs=num_correction_pairs,
            tolerance=gradient_tol,
        )
        self.set_trainable_parameters(results.position)
        self._logger.info(
            "Iterations: %s; Objective Evaluations: %s; Objective Value: %s; "
            "Converged: %s; Failed: %s",
            results.num_iterations.numpy(),
            results.num_objective_evaluations.numpy(),
            results.objective_value.numpy(),
            results.converged.numpy(),
            results.failed.numpy(),
        )

    def evaluate(self, data: DataSetIterator) -> Loss:
        """
        Evaluates the model by computing its mean physics-informed loss over
        the provided data set.

        :param data: the data set to evaluate the model on
        :return: the mean physics-informed loss
        """
        self._logger.debug("Evaluation")
        loss = self._compute_and_minimize_total_loss(data, optimizer=None)
        self._logger.debug("Total MSE - %s", loss)
        return loss

    def _create_diff_eq_lhs_functions(
        self,
    ) -> Sequence[PIDONSymbolMapFunction]:
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

    def _compute_and_minimize_total_loss(
        self,
        data: DataSetIterator,
        optimizer: Optional[tf.optimizers.Optimizer],
    ) -> Loss:
        """
        Computes the mean loss over the data set and, if an optimizer is
        provided, minimizes this loss via mini-batch gradient descent.

        :param data: the data set to compute the loss over
        :param optimizer: an optional optimizer instance; if one is provided,
            the model parameters are updated after each batch
        :return: the mean loss over the data set
        """
        batch_losses = []
        for batch_ind, batch in enumerate(data):
            batch_loss = self._compute_and_minimize_batch_loss(
                batch, optimizer
            )
            batch_losses.append(batch_loss)
            self._logger.debug(
                "Batch %s/%s MSE - %s", batch_ind + 1, len(data), batch_loss
            )

        return Loss.mean(
            batch_losses,
            self._diff_eq_loss_weights,
            self._ic_loss_weights,
            self._bc_loss_weights,
        )

    @tf.function
    def _compute_and_minimize_batch_loss(
        self, batch: DataBatch, optimizer: Optional[tf.optimizers.Optimizer]
    ) -> Loss:
        """
        Performs a forward pass over the batch, computes the batch loss, and
        if an optimizer is provided, updates the model parameters.

        :param batch: the batch to compute the losses over
        :param optimizer: the optimizer to use to update parameters of the
            model; if it is None, the model's parameters are not updated
        :return: the mean loss over the batch
        """
        if not optimizer:
            return self._compute_batch_loss(batch, False)

        with AutoDifferentiator() as auto_diff:
            loss = self._compute_batch_loss(batch, True)

        optimizer.minimize(
            loss.weighted_total_loss, self.trainable_variables, tape=auto_diff
        )

        return loss

    def _compute_batch_loss(self, batch: DataBatch, training: bool) -> Loss:
        """
        Computes and returns the physics-informed loss over the batch
        consisting of the mean squared differential equation error, the mean
        squared initial condition error, in the case of PDEs, the mean
        squared Dirichlet and Neumann boundary condition errors, and if
        applicable, the regularization error.

        :param batch: the batch to compute the losses over
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean loss over the batch
        """
        domain_batch, initial_batch, boundary_batch = batch
        diff_eq_loss = self._compute_differential_equation_loss(
            domain_batch, training
        )
        ic_loss = self._compute_initial_condition_loss(initial_batch, training)
        bc_losses = (
            self._compute_boundary_condition_loss(boundary_batch, training)
            if boundary_batch
            else None
        )
        model_loss = (
            tf.reshape(tf.add_n(self.losses), (1,)) if self.losses else None
        )

        return Loss.construct(
            diff_eq_loss,
            ic_loss,
            bc_losses,
            model_loss,
            self._diff_eq_loss_weights,
            self._ic_loss_weights,
            self._bc_loss_weights,
        )

    def _compute_differential_equation_loss(
        self, batch: DomainDataBatch, training: bool
    ) -> tf.Tensor:
        """
        Computes and returns the mean squared differential equation error.

        :param batch: the domain data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared differential equation error
        """
        with AutoDifferentiator(persistent=True) as auto_diff:
            auto_diff.watch(batch.t)
            if batch.x is not None:
                auto_diff.watch(batch.x)

            y_hat = self.__call__(
                (batch.u, batch.t, batch.x), training=training
            )

            symbol_map_arg = PIDONSymbolMapArg(
                auto_diff, batch.t, batch.x, y_hat
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
        self, batch: InitialDataBatch, training: bool
    ) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param batch: the initial condition data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared initial condition error
        """
        y_hat = self.__call__((batch.u, batch.t, batch.x), training=training)
        squared_ic_error = tf.square(y_hat - batch.y)
        return tf.reduce_mean(squared_ic_error, axis=0)

    def _compute_boundary_condition_loss(
        self, batch: BoundaryDataBatch, training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the mean squared Dirichlet boundary condition
        error and the mean squared Neumann boundary condition error.

        :param batch: the boundary data batch
        :param training: whether to call the underlying DeepONet in training
            mode
        :return: the mean squared Dirichlet and Neumann boundary condition
            errors
        """
        with AutoDifferentiator() as auto_diff:
            auto_diff.watch(batch.x)
            y_hat = self.__call__(
                (batch.u, batch.t, batch.x), training=training
            )

        d_y_over_d_n_hat = auto_diff.batch_gradient(batch.x, y_hat, batch.axes)

        dirichlet_bc_error = y_hat - batch.y
        dirichlet_bc_error = tf.where(
            tf.math.is_nan(batch.y), tf.zeros_like(batch.y), dirichlet_bc_error
        )
        squared_dirichlet_bc_error = tf.square(dirichlet_bc_error)
        mean_squared_dirichlet_bc_error = tf.reduce_mean(
            squared_dirichlet_bc_error, axis=0
        )

        neumann_bc_error = d_y_over_d_n_hat - batch.d_y_over_d_n
        neumann_bc_error = tf.where(
            tf.math.is_nan(batch.d_y_over_d_n),
            tf.zeros_like(batch.d_y_over_d_n),
            neumann_bc_error,
        )
        squared_neumann_bc_error = tf.square(neumann_bc_error)
        mean_squared_neumann_bc_error = tf.reduce_mean(
            squared_neumann_bc_error, axis=0
        )

        return mean_squared_dirichlet_bc_error, mean_squared_neumann_bc_error
