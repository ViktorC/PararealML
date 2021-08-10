from typing import Optional, Sequence, Dict, Any, Union, Tuple, NamedTuple, \
    List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs
from pararealml.core.operators.pidon.auto_differentiator import \
    AutoDifferentiator
from pararealml.core.operators.pidon.data_set import DataSetIterator, DataBatch
from pararealml.core.operators.pidon.pidon_symbol_mapper import \
    PIDONSymbolMapper, PIDONSymbolMapArg, PIDONSymbolMapFunction


class LossTensors(NamedTuple):
    """
    A collection of the various losses of a physics-informed DeepONet in
    tensor form.
    """
    total_weighted_loss: tf.Tensor
    diff_eq_loss: tf.Tensor
    ic_loss: tf.Tensor
    dirichlet_bc_loss: tf.Tensor
    neumann_bc_loss: tf.Tensor


class PIDeepONet:
    """
    A Physics Informed DeepONet model.

    See: https://arxiv.org/abs/2103.10974
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            latent_output_size: int,
            branch_net_hidden_layer_sizes: Optional[List[int]] = None,
            trunk_net_hidden_layer_sizes: Optional[List[int]] = None,
            activation: str = 'relu',
            initialisation: str = 'he_normal'
    ):
        """
        :param cp: the constrained problem to build a physics informed neural
            network around
        :param latent_output_size: the size of the latent output of the
            DeepONet model
        :param branch_net_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the branch net
        :param trunk_net_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the trunk net
        :param activation: the activation function to use for the hidden layers
        :param initialisation: the initialisation method to use for the weights
        """
        diff_eq = cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        self._cp = cp
        self._latent_output_size = latent_output_size

        if x_dimension:
            mesh = cp.mesh
            self._sensor_points = mesh.all_x(False)
            sensor_input_size = self._sensor_points.shape[0] * y_dimension
        else:
            self._sensor_points = None
            sensor_input_size = y_dimension

        if branch_net_hidden_layer_sizes is None:
            branch_net_hidden_layer_sizes = []
        if trunk_net_hidden_layer_sizes is None:
            trunk_net_hidden_layer_sizes = []

        self._branch_net = create_fnn(
            [sensor_input_size] +
            branch_net_hidden_layer_sizes +
            [latent_output_size * y_dimension],
            activation,
            initialisation)
        self._trunk_net = create_fnn(
            [x_dimension + 1] +
            trunk_net_hidden_layer_sizes +
            [latent_output_size * y_dimension],
            activation,
            initialisation)

        self._differentiator = AutoDifferentiator()

        self._symbol_mapper = PIDONSymbolMapper(cp, self._differentiator)
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
        Initialises the branch and trunk networks.
        """
        self._branch_net.build()
        self._trunk_net.build()

    def train(
            self,
            epochs: int,
            optimizer: Union[str, Dict[str, Any]],
            training_data: DataSetIterator,
            test_data: Optional[DataSetIterator] = None,
            diff_eq_loss_weight: float = 1,
            ic_loss_weight: float = 1,
            bc_loss_weight: float = 1,
            verbose: bool = True
    ) -> Tuple[Sequence[LossTensors], Sequence[LossTensors]]:
        """
        Trains the branch and trunk net parameters by minimising the physics
        informed loss function.

        :param epochs: the number of epochs over the training data
        :param optimizer: the optimizer to use to minimize the loss function
        :param training_data: the data set providing the training mini batches
        :param test_data: the data set providing the test mini batches
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
        :param verbose: whether loss information should be periodically printed
            to the console
        :return: the training and test loss histories
        """
        if epochs < 1:
            raise ValueError

        optimizer_instance = tf.keras.optimizers.get(optimizer)

        training_loss_history = []
        test_loss_history = []
        for i in range(epochs):
            training_batch_losses = []
            for batch in training_data:
                with tf.GradientTape(persistent=True) as tape:
                    self._differentiator.tape = tape
                    loss = self._batch_loss(
                        batch,
                        diff_eq_loss_weight,
                        ic_loss_weight,
                        bc_loss_weight)

                training_batch_losses.append(loss)
                optimizer_instance.minimize(
                    loss.total_weighted_loss,
                    self._branch_net.trainable_variables +
                    self._trunk_net.trainable_variables,
                    tape=tape)

            training_epoch_loss = average_loss_tensors(training_batch_losses)
            training_loss_history.append(training_epoch_loss)
            training_data.reset()

            if test_data:
                test_batch_losses = []
                for batch in test_data:
                    with tf.GradientTape(persistent=True) as tape:
                        self._differentiator.tape = tape
                        loss = self._batch_loss(
                            batch,
                            diff_eq_loss_weight,
                            ic_loss_weight,
                            bc_loss_weight)

                    test_batch_losses.append(loss)

                test_epoch_loss = average_loss_tensors(test_batch_losses)
                test_loss_history.append(test_epoch_loss)
                test_data.reset()
            else:
                test_epoch_loss = None

            if verbose:
                print('Epoch: ', i)
                print('DE Loss:',
                      'Training -', training_epoch_loss.diff_eq_loss,
                      'Test -', None if test_epoch_loss is None
                      else test_epoch_loss.diff_eq_loss)
                print('IC Loss:',
                      'Training -', training_epoch_loss.ic_loss,
                      'Test -', None if test_epoch_loss is None
                      else test_epoch_loss.ic_loss)
                print('Dirichlet BC Loss:',
                      'Training -', training_epoch_loss.dirichlet_bc_loss,
                      'Test -', None if test_epoch_loss is None
                      else test_epoch_loss.dirichlet_bc_loss)
                print('Neumann BC Loss:',
                      'Training -', training_epoch_loss.neumann_bc_loss,
                      'Test -', None if test_epoch_loss is None
                      else test_epoch_loss.neumann_bc_loss)
                print('Total Loss:',
                      'Training -', training_epoch_loss.total_weighted_loss,
                      'Test -', None if test_epoch_loss is None
                      else test_epoch_loss.total_weighted_loss)

        return training_loss_history, test_loss_history

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

        trunk_input = tf.concat([t, x], axis=1) if x_dimension else t

        branch_output = tf.reshape(
            self._branch_net(u),
            (-1, y_dimension, self._latent_output_size))
        trunk_output = tf.reshape(
            self._trunk_net(trunk_input),
            (-1, y_dimension, self._latent_output_size))
        combined_output = branch_output * trunk_output
        return tf.math.reduce_sum(combined_output, axis=-1)

    def _batch_loss(
            self,
            batch: DataBatch,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float
    ) -> LossTensors:
        """
        Computes all the losses over the batch.

        :param batch: the batch to compute the losses over
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
        :return: the various losses over the batch
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        domain_batch = batch.domain
        if domain_batch:
            diff_eq_loss = self._mean_squared_differential_equation_error(
                domain_batch.u, domain_batch.t, domain_batch.x)

            if x_dimension:
                sensor_x = tf.convert_to_tensor(
                    self._sensor_points, tf.float32)
                sensor_t = tf.zeros((sensor_x.shape[0], 1))
            else:
                sensor_x = None
                sensor_t = tf.zeros((1, 1))

            ic_loss = self._mean_squared_initial_condition_error(
                domain_batch.u, sensor_t, sensor_x)
        else:
            diff_eq_loss = tf.zeros((y_dimension,))
            ic_loss = tf.zeros((y_dimension,))

        boundary_batch = batch.boundary
        if boundary_batch:
            dirichlet_bc_loss, neumann_bc_loss = \
                self._mean_squared_boundary_condition_errors(
                    boundary_batch.u,
                    boundary_batch.t,
                    boundary_batch.y,
                    boundary_batch.d_y_over_d_n,
                    boundary_batch.axes)
        else:
            dirichlet_bc_loss = tf.zeros((1, y_dimension))
            neumann_bc_loss = tf.zeros((1, y_dimension))

        total_weighted_loss = \
            diff_eq_loss_weight * diff_eq_loss + \
            ic_loss_weight * ic_loss + \
            bc_loss_weight * (dirichlet_bc_loss + neumann_bc_loss)

        return LossTensors(
            total_weighted_loss,
            diff_eq_loss,
            ic_loss,
            dirichlet_bc_loss,
            neumann_bc_loss)

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
        self._differentiator.tape.watch(t)
        if x is not None:
            self._differentiator.tape.watch(x)

        y_hat = self.predict(u, t, x)

        symbol_map_arg = PIDONSymbolMapArg(t, x, y_hat)
        rhs = self._symbol_mapper.map(symbol_map_arg)

        diff_eq_residual = []
        for i in range(len(rhs)):
            lhs = self._diff_eq_lhs_functions[i](symbol_map_arg)
            rhs = rhs[i]
            diff_eq_residual.append(lhs - rhs)

        squared_diff_eq_error = tf.square(tf.concat(diff_eq_residual, axis=1))
        return tf.reduce_mean(squared_diff_eq_error, axis=0)

    @tf.function
    def _mean_squared_initial_condition_error(
            self,
            u: tf.Tensor,
            t: tf.Tensor,
            x: Optional[tf.Tensor]) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param u: the initial condition sensor readings
        :param t: the sensor time points
        :param x: the sensor space points; if the IVP is based on an ODE, None
        :return: the mean squared initial condition error
        """
        y_hat = tf.stack([
            self.predict(
                tf.tile(u[i:i + 1, :], (t.shape[0], 1)),
                t,
                x
            ) for i in range(u.shape[0])
        ], axis=0)

        squared_ic_error = tf.reduce_mean(
            tf.square(tf.reshape(y_hat, u.shape) - u),
            axis=1,
            keepdims=True)
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
        self._differentiator.tape.watch(x)

        y_hat = self.predict(u, t, x)
        d_y_over_d_n_hat = self._differentiator.gradient(x, y_hat, axes)

        dirichlet_bc_error = y_hat - y
        dirichlet_bc_error = tf.where(
            tf.math.is_nan(dirichlet_bc_error),
            tf.zeros_like(dirichlet_bc_error),
            dirichlet_bc_error)
        squared_dirichlet_bc_error = tf.square(dirichlet_bc_error)
        mean_squared_dirichlet_bc_error = \
            tf.reduce_mean(squared_dirichlet_bc_error, axis=0)

        neumann_bc_error = d_y_over_d_n_hat - d_y_over_d_n
        neumann_bc_error = tf.where(
            tf.math.is_nan(neumann_bc_error),
            tf.zeros_like(neumann_bc_error),
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
                    lambda arg, _y_ind=y_ind:
                    self._differentiator.gradient(
                        arg.t, arg.y_hat[:, _y_ind:_y_ind + 1], 0))
            elif lhs_type == Lhs.Y:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind: arg.y_hat[:, _y_ind:_y_ind + 1])
            elif lhs_type == Lhs.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda arg, _y_ind=y_ind:
                    self._differentiator.laplacian(
                        arg.x,
                        arg.y_hat[:, _y_ind:_y_ind + 1],
                        self._cp.mesh.coordinate_system_type))
            else:
                raise ValueError

        return lhs_functions


def create_fnn(
        layer_sizes: Sequence[int],
        activation: str,
        initialisation: str
) -> Sequential:
    """
    Creates a fully-connected neural network model.

    :param layer_sizes: a list of the sizes of the input, hidden, and output
        layers
    :param activation: the activation function to use for the hidden layers
    :param initialisation: the initialisation method to use for the weights
    :return: the fully-connected neural network model
    """
    if len(layer_sizes) < 2:
        raise ValueError

    model = Sequential()
    model.add(InputLayer(input_shape=layer_sizes[0]))
    for layer_size in layer_sizes[1:-1]:
        model.add(Dense(
            layer_size,
            activation=activation,
            kernel_initializer=initialisation))
    model.add(Dense(layer_sizes[-1], kernel_initializer=initialisation))
    return model


def average_loss_tensors(losses: Sequence[LossTensors]) -> LossTensors:
    """
    Computes the average of the provided loss tensors.

    :param losses: the losses to average over
    :return: the mean loss tensors
    """
    total_weighted_losses = []
    diff_eq_losses = []
    ic_losses = []
    dirichlet_bc_losses = []
    neumann_bc_losses = []
    for loss_tensor in losses:
        total_weighted_losses.append(loss_tensor.total_weighted_loss)
        diff_eq_losses.append(loss_tensor.diff_eq_loss)
        ic_losses.append(loss_tensor.ic_loss)
        dirichlet_bc_losses.append(loss_tensor.dirichlet_bc_loss)
        neumann_bc_losses.append(loss_tensor.neumann_bc_loss)

    return LossTensors(
        tf.reduce_mean(tf.stack(total_weighted_losses), axis=0),
        tf.reduce_mean(tf.stack(diff_eq_losses), axis=0),
        tf.reduce_mean(tf.stack(ic_losses), axis=0),
        tf.reduce_mean(tf.stack(dirichlet_bc_losses), axis=0),
        tf.reduce_mean(tf.stack(neumann_bc_losses), axis=0))
