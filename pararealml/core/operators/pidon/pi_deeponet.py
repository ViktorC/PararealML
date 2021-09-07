from typing import Optional, Sequence, Dict, Any, Union, Tuple, List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Optimizer

import tensorflow_probability as tfp

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs
from pararealml.core.operators.pidon.auto_differentiator import \
    AutoDifferentiator
from pararealml.core.operators.pidon.data_set import DataSetIterator, DataBatch
from pararealml.core.operators.pidon.loss import Loss
from pararealml.core.operators.pidon.pidon_symbol_mapper import \
    PIDONSymbolMapper, PIDONSymbolMapArg, PIDONSymbolMapFunction


class PIDeepONet:
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
            branch_initialization: Optional[str] = None,
            trunk_initialization: Optional[str] = None,
            branch_activation: Optional[str] = 'tanh',
            trunk_activation: Optional[str] = 'tanh'
    ):
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
        """
        if latent_output_size < 1:
            raise ValueError(
                f'latent output size ({latent_output_size}) must be greater '
                f'than 0')

        diff_eq = cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        self._cp = cp
        self._latent_output_size = latent_output_size

        if x_dimension:
            mesh = cp.mesh
            self._sensor_points = mesh.all_index_coordinates(
                False, flatten=True)
            sensor_input_size = self._sensor_points.shape[0] * y_dimension
        else:
            self._sensor_points = None
            sensor_input_size = y_dimension

        if branch_hidden_layer_sizes is None:
            branch_hidden_layer_sizes = []
        if trunk_hidden_layer_sizes is None:
            trunk_hidden_layer_sizes = []

        self._branch_net = create_regression_fnn(
            [sensor_input_size] +
            branch_hidden_layer_sizes +
            [latent_output_size * y_dimension],
            branch_initialization,
            branch_activation)
        self._trunk_net = create_regression_fnn(
            [x_dimension + 1] +
            trunk_hidden_layer_sizes +
            [latent_output_size * y_dimension],
            trunk_initialization,
            trunk_activation)

        self._symbol_mapper = PIDONSymbolMapper(cp)
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the model is built around.
        """
        return self._cp

    @property
    def branch_net(self) -> Sequential:
        """
        The branch neural network of the model.
        """
        return self._branch_net

    @property
    def trunk_net(self) -> Sequential:
        """
        The trunk neural network of the model.
        """
        return self._trunk_net

    def init(self):
        """
        Initializes the branch and trunk networks.
        """
        self._branch_net.build()
        self._trunk_net.build()

    def fit(
            self,
            epochs: int,
            optimizer: Union[str, Dict[str, Any]],
            training_data: DataSetIterator,
            test_data: Optional[DataSetIterator] = None,
            diff_eq_loss_weight: float = 1.,
            ic_loss_weight: float = 1.,
            bc_loss_weight: float = 1.,
            verbose: bool = True
    ) -> Tuple[List[Loss], List[Loss]]:
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
            raise ValueError(
                f'number of epochs ({epochs}) must be greater than 0')

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
                test_loss_history.append(training_epoch_loss)

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
            for var in self._branch_net.trainable_variables + \
                    self._trunk_net.trainable_variables:
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

            gradients = auto_diff.gradient(
                value,
                self._branch_net.trainable_variables +
                self._trunk_net.trainable_variables)
            flattened_gradients = tf.concat([
                tf.reshape(gradient, (1, -1)) for gradient in gradients
            ], axis=1)

            return value, flattened_gradients

        if verbose:
            print('L-BFGS Optimization')

        flattened_parameters = tf.concat([
            tf.reshape(var, (1, -1)) for var in
            self._branch_net.trainable_variables +
            self._trunk_net.trainable_variables
        ], axis=1)
        results = tfp.optimizer.lbfgs_minimize(
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

        if test_data:
            test_loss = self._physics_informed_loss(
                test_data.get_full_batch(),
                diff_eq_loss_weight,
                ic_loss_weight,
                bc_loss_weight)
            print('Test MSE -', test_loss)
        else:
            test_loss = None

        return training_loss, test_loss

    @tf.function
    def predict(
            self,
            u: tf.Tensor,
            t: tf.Tensor,
            x: Optional[tf.Tensor]) -> tf.Tensor:
        """
        Predicts (y ∘ u)(t, x).

        :param u: sensor readings of the value of the function u at a set of
            points
        :param t: the temporal input variable of the composed function
        :param x: the spatial input variables of the composed function
        :return: the predicted value of (y ∘ u)(t, x)
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        branch_output = tf.reshape(
            self._branch_net(u),
            (-1, self._latent_output_size, y_dimension))
        trunk_output = tf.reshape(
            self._trunk_net(tf.concat([t, x], axis=1) if x_dimension else t),
            (-1, self._latent_output_size, y_dimension))
        return tf.math.reduce_sum(branch_output * trunk_output, axis=1)

    @tf.function
    def _step(
            self,
            batch: DataBatch,
            optimizer: Optimizer,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float
    ) -> Loss:
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
            self._branch_net.trainable_variables +
            self._trunk_net.trainable_variables,
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
        domain_batch, boundary_batch = batch

        diff_eq_loss = self._mean_squared_differential_equation_error(
            domain_batch.u, domain_batch.t, domain_batch.x)
        ic_loss = self._mean_squared_initial_condition_error(
            domain_batch.u)
        bc_losses = self._mean_squared_boundary_condition_errors(
            boundary_batch.u,
            boundary_batch.t,
            boundary_batch.x,
            boundary_batch.y,
            boundary_batch.d_y_over_d_n,
            boundary_batch.axes) if boundary_batch else None

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
            u: tf.Tensor,
            t: tf.Tensor,
            x: Optional[tf.Tensor]) -> tf.Tensor:
        """
        Computes and returns the mean squared differential equation error.

        :param u: the initial condition sensor readings
        :param t: the time points
        :param x: the space points; if the IVP is based on an ODE, None
        :return: the mean squared differential equation error
        """
        with AutoDifferentiator(persistent=True) as auto_diff:
            auto_diff.watch(t)
            if x is not None:
                auto_diff.watch(x)

            y_hat = self.predict(u, t, x)

            symbol_map_arg = PIDONSymbolMapArg(auto_diff, t, x, y_hat)
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
            u: tf.Tensor) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param u: the initial condition sensor readings
        :return: the mean squared initial condition error
        """
        if self._cp.differential_equation.x_dimension:
            t = tf.zeros((self._sensor_points.shape[0], 1), dtype=tf.float32)
            x = tf.convert_to_tensor(self._sensor_points, dtype=tf.float32)
        else:
            t = tf.zeros((1, 1), dtype=tf.float32)
            x = None

        y_hat = tf.map_fn(
            fn=lambda u_i: self.predict(
                tf.tile(tf.reshape(u_i, (1, -1)), (t.shape[0], 1)), t, x),
            elems=u)

        squared_ic_error = tf.reduce_mean(
            tf.square(y_hat - tf.reshape(u, y_hat.shape)), axis=1)
        return tf.reduce_mean(squared_ic_error, axis=0)

    @tf.function
    def _mean_squared_boundary_condition_errors(
            self,
            u: tf.Tensor,
            t: tf.Tensor,
            x: tf.Tensor,
            y: tf.Tensor,
            d_y_over_d_n: tf.Tensor,
            axes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the mean squared Dirichlet boundary condition
        error and the mean squared Neumann boundary condition error.

        :param u: the initial condition sensor readings
        :param t: the time points
        :param x: the space points
        :param y: the expected boundary values
        :param d_y_over_d_n: the expected directional derivatives of the
            unknown function in the direction of the normal vector of the
            boundary
        :param axes: the indices of the axes orthogonal to the boundary
        :return: the mean squared Dirichlet and Neumann boundary condition
            errors
        """
        with AutoDifferentiator() as auto_diff:
            auto_diff.watch(x)
            y_hat = self.predict(u, t, x)

        d_y_over_d_n_hat = auto_diff.batch_gradient(x, y_hat, axes)

        dirichlet_bc_error = y_hat - y
        dirichlet_bc_error = tf.where(
            tf.math.is_nan(y),
            tf.zeros_like(y),
            dirichlet_bc_error)
        squared_dirichlet_bc_error = tf.square(dirichlet_bc_error)
        mean_squared_dirichlet_bc_error = \
            tf.reduce_mean(squared_dirichlet_bc_error, axis=0)

        neumann_bc_error = d_y_over_d_n_hat - d_y_over_d_n
        neumann_bc_error = tf.where(
            tf.math.is_nan(d_y_over_d_n),
            tf.zeros_like(d_y_over_d_n),
            neumann_bc_error)
        squared_neumann_bc_error = tf.square(neumann_bc_error)
        mean_squared_neumann_bc_error = \
            tf.reduce_mean(squared_neumann_bc_error, axis=0)

        return mean_squared_dirichlet_bc_error, mean_squared_neumann_bc_error

    def _create_diff_eq_lhs_functions(
            self
    ) -> Sequence[PIDONSymbolMapFunction]:
        """
        Creates a sequence of symbol map functions representing the left hand
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
                    f'unsupported left hand side type ({lhs_type.name})')

        return lhs_functions


def create_regression_fnn(
        layer_sizes: Sequence[int],
        initialization: Optional[str] = None,
        activation: Optional[str] = None
) -> Sequential:
    """
    Creates a fully-connected feedforward neural network regression model.

    :param layer_sizes: a list of the sizes of the layers including the input
        layer
    :param initialization: the initialization method to use for the weights of
        the layers
    :param activation: the activation function to use for the hidden layers
    :return: the fully-connected neural network model
    """
    if len(layer_sizes) < 2:
        raise ValueError(
            f'number of layers ({len(layer_sizes)}) must be greater than 1')

    if initialization is None:
        initialization = 'glorot_uniform'

    model = Sequential()
    model.add(InputLayer(input_shape=layer_sizes[0]))
    for layer_size in layer_sizes[1:-1]:
        model.add(Dense(
            layer_size,
            kernel_initializer=initialization,
            activation=activation))
    model.add(Dense(layer_sizes[-1], kernel_initializer=initialization))

    return model
