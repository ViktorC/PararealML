from typing import Optional, Sequence, Dict, Any, Union, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow_probability.python.optimizer import lbfgs_minimize

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import Lhs
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.pidon.auto_differentiator import \
    AutoDifferentiator
from pararealml.operators.ml.pidon.data_set import DataSetIterator, \
    DataBatch, InitialDataBatch, BoundaryDataBatch, DomainDataBatch
from pararealml.operators.ml.pidon.loss import Loss
from pararealml.operators.ml.pidon.pidon_symbol_mapper import \
    PIDONSymbolMapper, PIDONSymbolMapArg, PIDONSymbolMapFunction


class PIDeepONet(DeepONet):
    """
    A Physics-Informed DeepONet model.

    See: https://arxiv.org/abs/2103.10974
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            latent_output_size: int,
            branch_hidden_layer_sizes: Optional[List[int]] = None,
            trunk_hidden_layer_sizes: Optional[List[int]] = None,
            branch_initialization: str = 'glorot_uniform',
            trunk_initialization: str = 'glorot_uniform',
            branch_activation: Optional[str] = 'tanh',
            trunk_activation: Optional[str] = 'tanh',
            vertex_oriented: bool = False):
        """
        :param cp: the constrained problem to build a physics-informed neural
            network around
        :param latent_output_size: the size of the latent output of the
            branch and trunk networks of the DeepONet model
        :param branch_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the branch net
        :param trunk_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the trunk net
        :param branch_initialization: the initialization method to use for the
            weights of the branch net
        :param trunk_initialization: the initialization method to use for the
            weights of the trunk net
        :param branch_activation: the activation function to use for the layers
            of the branch net
        :param trunk_activation: the activation function to use for the layers
            of the trunk net
        :param vertex_oriented: whether the initial condition collocation
            points are the vertices or the cell centers of the mesh
        """
        if latent_output_size < 1:
            raise ValueError('latent output size must be greater than 0')

        diff_eq = cp.differential_equation
        x_dim = diff_eq.x_dimension
        y_dim = diff_eq.y_dimension

        self._cp = cp
        self._latent_output_size = latent_output_size

        if branch_hidden_layer_sizes is None:
            branch_hidden_layer_sizes = []
        if trunk_hidden_layer_sizes is None:
            trunk_hidden_layer_sizes = []

        super(PIDeepONet, self).__init__(
            [
                np.prod(cp.mesh.shape(vertex_oriented)).item() * y_dim
                if x_dim else y_dim
            ] +
            branch_hidden_layer_sizes +
            [latent_output_size * y_dim],
            [x_dim + 1] +
            trunk_hidden_layer_sizes +
            [latent_output_size * y_dim],
            y_dim,
            branch_initialization=branch_initialization,
            branch_activation=branch_activation,
            trunk_initialization=trunk_initialization,
            trunk_activation=trunk_activation)

        self._symbol_mapper = PIDONSymbolMapper(cp)
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the model is built around.
        """
        return self._cp

    def fit(
            self,
            epochs: int,
            optimizer: Union[str, Dict[str, Any], Optimizer],
            training_data: DataSetIterator,
            test_data: Optional[DataSetIterator] = None,
            diff_eq_loss_weight: float = 1.,
            ic_loss_weight: float = 1.,
            bc_loss_weight: float = 1.,
            verbose: bool = True) -> Tuple[List[Loss], List[Loss]]:
        """
        Fits the branch and trunk net parameters by minimising the
        physics-informed loss function over the provided training data set. It
        also evaluates the loss over both the training data and the test data,
        if provided, for every epoch.

        :param epochs: the number of epochs over the training data
        :param optimizer: the optimizer to use to minimize the loss function
        :param training_data: the data set providing the training mini batches
        :param test_data: the data set providing the test mini batches
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :param verbose: whether loss information should be periodically printed
            to the console
        :return: the training and test loss histories
        """
        if epochs < 1:
            raise ValueError('number of epochs must be greater than 0')

        optimizer_instance = tf.keras.optimizers.get(optimizer)

        if verbose:
            print('Gradient Descent Optimization')

        training_loss_history = []
        test_loss_history = []
        for i in range(epochs):
            if verbose:
                print('Epoch:', i)

            training_batch_losses = []
            for batch in training_data:
                training_batch_losses.append(self._step(
                    batch,
                    optimizer_instance,
                    diff_eq_loss_weight,
                    ic_loss_weight,
                    bc_loss_weight))
            training_epoch_loss = Loss.mean(
                training_batch_losses,
                diff_eq_loss_weight,
                ic_loss_weight,
                bc_loss_weight)
            training_loss_history.append(training_epoch_loss)
            if verbose:
                print('Training MSE -', training_epoch_loss)

            if test_data:
                test_batch_losses = []
                for batch in test_data:
                    test_batch_losses.append(self._physics_informed_loss(
                        batch,
                        diff_eq_loss_weight,
                        ic_loss_weight,
                        bc_loss_weight))
                test_epoch_loss = Loss.mean(
                    test_batch_losses,
                    diff_eq_loss_weight,
                    ic_loss_weight,
                    bc_loss_weight)
                test_loss_history.append(test_epoch_loss)
                if verbose:
                    print('Test MSE -', test_epoch_loss)

        return training_loss_history, test_loss_history

    def fit_with_lbfgs(
            self,
            training_data: DataSetIterator,
            max_iterations: int,
            gradient_tol: float,
            test_data: Optional[DataSetIterator] = None,
            diff_eq_loss_weight: float = 1.,
            ic_loss_weight: float = 1.,
            bc_loss_weight: float = 1.,
            verbose: bool = True) -> Tuple[Loss, Optional[Loss]]:
        """
        Fits the branch and trunk net parameters by minimising the
        physics-informed loss function over the provided training data set
        using the L-BFGS optimization method. It also evaluates the loss over
        both the training data and the test data, if provided, for every epoch.

        :param training_data: the data set providing the full training batch
        :param max_iterations: the maximum number of iterations to perform the
            optimization for
        :param gradient_tol: the threshold on the gradient vector; if the
            largest element of the absolute value of the gradient vector is
            less than this threshold, the optimization is stopped
        :param test_data: the data set providing the full test batch
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :param verbose: whether loss information should be periodically printed
            to the console
        :return: the training and test loss histories
        """
        full_training_data_batch = training_data.get_full_batch()

        @tf.function
        def set_model_parameters(parameters: tf.Tensor):
            offset = 0
            for var in self.trainable_variables:
                var_size = tf.reduce_prod(var.shape)
                var.assign(tf.reshape(
                    parameters[0, offset:offset + var_size], var.shape))
                offset += var_size

        @tf.function
        def value_and_gradients_function(
                parameters: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            set_model_parameters(parameters)
            with AutoDifferentiator() as auto_diff:
                loss = self._physics_informed_loss(
                    full_training_data_batch,
                    diff_eq_loss_weight,
                    ic_loss_weight,
                    bc_loss_weight)
                value = tf.reduce_sum(loss.weighted_total_loss, keepdims=True)

            gradients = auto_diff.gradient(value, self.trainable_variables)
            flattened_gradients = tf.concat(
                [tf.reshape(gradient, (1, -1)) for gradient in gradients],
                axis=1)
            return value, flattened_gradients

        if verbose:
            print('L-BFGS Optimization')

        flattened_parameters = tf.concat(
            [tf.reshape(var, (1, -1)) for var in self.trainable_variables],
            axis=1)
        results = lbfgs_minimize(
            value_and_gradients_function=value_and_gradients_function,
            initial_position=flattened_parameters,
            max_iterations=max_iterations,
            tolerance=gradient_tol)
        set_model_parameters(results.position)

        training_loss = self._physics_informed_loss(
            full_training_data_batch,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)
        if verbose:
            print('Training MSE -', training_loss)

        test_loss = None
        if test_data:
            test_loss = self._physics_informed_loss(
                test_data.get_full_batch(),
                diff_eq_loss_weight,
                ic_loss_weight,
                bc_loss_weight)
            print('Test MSE -', test_loss)

        return training_loss, test_loss

    @tf.function
    def _step(
            self,
            batch: DataBatch,
            optimizer: Optimizer,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float) -> Loss:
        """
        Performs a forward pass on the batch, computes the batch loss, and
        updates the model parameters.

        :param batch: the batch to compute the losses over
        :param optimizer: the optimizer to use to update parameters of the
            model
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :return: the various losses over the batch
        """
        with AutoDifferentiator() as auto_diff:
            loss = self._physics_informed_loss(
                batch, diff_eq_loss_weight, ic_loss_weight, bc_loss_weight)

        optimizer.minimize(
            loss.weighted_total_loss,
            self.trainable_variables,
            tape=auto_diff)

        return loss

    @tf.function
    def _physics_informed_loss(
            self,
            batch: DataBatch,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float) -> Loss:
        """
        Computes and returns the total physics-informed loss over the batch
        consisting of the mean squared differential equation error, the mean
        squared initial condition error, and in the case of PDEs, the mean
        squared Dirichlet and Neumann boundary condition errors.

        :param batch: the batch to compute the losses over
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :return: the total physics-informed loss over the batch
        """
        domain_batch, initial_batch, boundary_batch = batch
        diff_eq_loss = \
            self._mean_squared_differential_equation_error(domain_batch)
        ic_loss = self._mean_squared_initial_condition_error(initial_batch)
        bc_losses = \
            self._mean_squared_boundary_condition_errors(boundary_batch) \
            if boundary_batch else None

        return Loss.construct(
            diff_eq_loss,
            ic_loss,
            bc_losses,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)

    @tf.function
    def _mean_squared_differential_equation_error(
            self,
            batch: DomainDataBatch) -> tf.Tensor:
        """
        Computes and returns the mean squared differential equation error.

        :param batch: the domain data batch
        :return: the mean squared differential equation error
        """
        with AutoDifferentiator(persistent=True) as auto_diff:
            auto_diff.watch(batch.t)
            if batch.x is not None:
                auto_diff.watch(batch.x)

            y_hat = self.call((batch.u, batch.t, batch.x))

            symbol_map_arg = PIDONSymbolMapArg(
                auto_diff, batch.t, batch.x, y_hat)
            rhs = self._symbol_mapper.map(symbol_map_arg)

            diff_eq_residual = tf.concat([
                self._diff_eq_lhs_functions[i](symbol_map_arg) - rhs[i]
                for i in range(len(rhs))
            ], axis=1)

        squared_diff_eq_error = tf.square(diff_eq_residual)
        return tf.reduce_mean(squared_diff_eq_error, axis=0)

    @tf.function
    def _mean_squared_initial_condition_error(
            self,
            batch: InitialDataBatch) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param batch: the initial condition data batch
        :return: the mean squared initial condition error
        """
        y_hat = self.call((batch.u, batch.t, batch.x))
        squared_ic_error = tf.square(y_hat - batch.y)
        return tf.reduce_mean(squared_ic_error, axis=0)

    @tf.function
    def _mean_squared_boundary_condition_errors(
            self,
            batch: BoundaryDataBatch) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the mean squared Dirichlet boundary condition
        error and the mean squared Neumann boundary condition error.

        :param batch: the boundary data batch
        :return: the mean squared Dirichlet and Neumann boundary condition
            errors
        """
        with AutoDifferentiator() as auto_diff:
            auto_diff.watch(batch.x)
            y_hat = self.call((batch.u, batch.t, batch.x))

        d_y_over_d_n_hat = auto_diff.batch_gradient(batch.x, y_hat, batch.axes)

        dirichlet_bc_error = y_hat - batch.y
        dirichlet_bc_error = tf.where(
            tf.math.is_nan(batch.y),
            tf.zeros_like(batch.y),
            dirichlet_bc_error)
        squared_dirichlet_bc_error = tf.square(dirichlet_bc_error)
        mean_squared_dirichlet_bc_error = \
            tf.reduce_mean(squared_dirichlet_bc_error, axis=0)

        neumann_bc_error = d_y_over_d_n_hat - batch.d_y_over_d_n
        neumann_bc_error = tf.where(
            tf.math.is_nan(batch.d_y_over_d_n),
            tf.zeros_like(batch.d_y_over_d_n),
            neumann_bc_error)
        squared_neumann_bc_error = tf.square(neumann_bc_error)
        mean_squared_neumann_bc_error = \
            tf.reduce_mean(squared_neumann_bc_error, axis=0)

        return mean_squared_dirichlet_bc_error, mean_squared_neumann_bc_error

    def _create_diff_eq_lhs_functions(
            self) -> Sequence[PIDONSymbolMapFunction]:
        """
        Creates a sequence of symbol map functions representing the left-hand
        side of the differential equation system.
        """
        diff_eq = self._cp.differential_equation

        lhs_functions = []
        for y_ind, lhs_type in \
                enumerate(diff_eq.symbolic_equation_system.lhs_types):
            if lhs_type == Lhs.D_Y_OVER_D_T:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.auto_diff.batch_gradient(
                        arg.t, arg.y_hat[:, _y_ind:_y_ind + 1], 0))

            elif lhs_type == Lhs.Y:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.y_hat[:, _y_ind:_y_ind + 1])

            elif lhs_type == Lhs.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.auto_diff.batch_laplacian(
                        arg.x,
                        arg.y_hat[:, _y_ind:_y_ind + 1],
                        self._cp.mesh.coordinate_system_type))

            else:
                raise ValueError(
                    f'unsupported left-hand side type ({lhs_type.name})')

        return lhs_functions
