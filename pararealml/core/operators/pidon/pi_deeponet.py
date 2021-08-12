from typing import Optional, Sequence, Dict, Any, Union, Tuple, List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Optimizer

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
    A Physics Informed DeepONet model.

    See: https://arxiv.org/abs/2103.10974
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            branch_net_layer_sizes: List[int],
            trunk_net_layer_sizes: List[int],
            branch_initialisation: Optional[str] = None,
            trunk_initialisation: Optional[str] = None,
            branch_activation: Optional[str] = 'tanh',
            trunk_activation: Optional[str] = 'tanh'
    ):
        """
        :param cp: the constrained problem to build a physics informed neural
            network around
        :param branch_net_layer_sizes: a list of the sizes of the hidden and
            output layers of the branch net
        :param trunk_net_layer_sizes: a list of the sizes of the hidden and
            output layers of the trunk net
        :param branch_initialisation: the initialisation method to use for the
            weights of the branch net
        :param trunk_initialisation: the initialisation method to use for the
            weights of the trunk net
        :param branch_activation: the activation function to use for the layers
            of the branch net
        :param trunk_activation: the activation function to use for the layers
            of the trunk net
        """
        if len(branch_net_layer_sizes) < 1:
            raise ValueError
        if len(trunk_net_layer_sizes) < 1:
            raise ValueError

        diff_eq = cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        self._cp = cp

        if x_dimension:
            mesh = cp.mesh
            self._sensor_points = mesh.all_x(False)
            sensor_input_size = self._sensor_points.shape[0] * y_dimension
        else:
            self._sensor_points = None
            sensor_input_size = y_dimension

        self._branch_net = create_fnn(
            [sensor_input_size] + branch_net_layer_sizes,
            branch_initialisation,
            branch_activation)
        self._trunk_net = create_fnn(
            [1 + x_dimension] + trunk_net_layer_sizes,
            trunk_initialisation,
            trunk_activation)
        self._combiner_net = create_fnn(
            [
                branch_net_layer_sizes[-1] + trunk_net_layer_sizes[-1],
                y_dimension
            ])

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
    def combiner_net(self) -> Sequential:
        """
        The combiner neural network of the model.
        """
        return self._combiner_net

    @property
    def trunk_net(self) -> Sequential:
        """
        The trunk neural network of the model.
        """
        return self._trunk_net

    def init(self):
        """
        Initialises the branch, trunk, and combiner networks.
        """
        self._branch_net.build()
        self._trunk_net.build()
        self._combiner_net.build()

    def train(
            self,
            epochs: int,
            optimizer: Union[str, Dict[str, Any]],
            training_data: DataSetIterator,
            test_data: Optional[DataSetIterator] = None,
            diff_eq_loss_weight: float = 1.,
            ic_loss_weight: float = 1.,
            bc_loss_weight: float = 1.,
            verbose: bool = True
    ) -> Tuple[Sequence[Loss], Sequence[Loss]]:
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
            if verbose:
                print('Epoch: ', i)

            training_epoch_loss = self._compute_epoch_loss(
                training_data,
                diff_eq_loss_weight,
                ic_loss_weight,
                bc_loss_weight,
                optimizer_instance)
            training_loss_history.append(training_epoch_loss)
            if verbose:
                print('Training MSE - ', training_epoch_loss)

            if test_data:
                test_epoch_loss = self._compute_epoch_loss(
                    test_data,
                    diff_eq_loss_weight,
                    ic_loss_weight,
                    bc_loss_weight,
                    None)
                test_loss_history.append(test_epoch_loss)
                if verbose:
                    print('Test MSE - ', test_epoch_loss)

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
        branch_output = self._branch_net(u)

        trunk_input = tf.concat([t, x], axis=1) \
            if self._cp.differential_equation.x_dimension else t
        trunk_output = self._trunk_net(trunk_input)

        combiner_input = tf.concat([branch_output, trunk_output], axis=1)
        combiner_output = self._combiner_net(combiner_input)

        return combiner_output

    def _compute_epoch_loss(
            self,
            data: DataSetIterator,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float,
            optimizer: Optional[Optimizer]) -> Loss:
        """
        Computes the mean epoch loss and if an optimizer is provided, it
        updates the parameters of the model after every batch as well.

        :param data: the data set iterator providing the epoch data
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
        :param optimizer: the optimizer to use to update parameters of the
            model
        :return: the mean losses over the epoch
        """
        batch_losses = []
        for batch in data:
            with tf.GradientTape(persistent=True) as tape:
                self._differentiator.tape = tape
                loss = self._compute_batch_loss(
                    batch,
                    diff_eq_loss_weight,
                    ic_loss_weight,
                    bc_loss_weight)

            batch_losses.append(loss)

            if optimizer is not None:
                optimizer.minimize(
                    loss.total_weighted_loss,
                    self._branch_net.trainable_variables +
                    self._trunk_net.trainable_variables +
                    self._combiner_net.trainable_variables,
                    tape=tape)

        data.reset()
        return Loss.mean(
            batch_losses,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)

    def _compute_batch_loss(
            self,
            batch: DataBatch,
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float
    ) -> Loss:
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
        domain_batch = batch.domain
        diff_eq_loss = self._compute_mean_squared_differential_equation_error(
            domain_batch.u, domain_batch.t, domain_batch.x)
        ic_loss = self._compute_mean_squared_initial_condition_error(
            domain_batch.u)

        boundary_batch = batch.boundary
        bc_losses = self._compute_mean_squared_boundary_condition_errors(
            boundary_batch.u,
            boundary_batch.t,
            boundary_batch.y,
            boundary_batch.d_y_over_d_n,
            boundary_batch.axes) if boundary_batch else None

        return Loss(
            diff_eq_loss,
            ic_loss,
            bc_losses,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)

    @tf.function
    def _compute_mean_squared_differential_equation_error(
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
    def _compute_mean_squared_initial_condition_error(
            self,
            u: tf.Tensor) -> tf.Tensor:
        """
        Computes and returns the mean squared initial condition error.

        :param u: the initial condition sensor readings
        :return: the mean squared initial condition error
        """
        if self._cp.differential_equation.x_dimension:
            x = tf.convert_to_tensor(self._sensor_points, tf.float32)
            t = tf.zeros((x.shape[0], 1))
        else:
            x = None
            t = tf.zeros((1, 1))

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
    def _compute_mean_squared_boundary_condition_errors(
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
        initialisation: Optional[str] = None,
        activation: Optional[str] = None
) -> Sequential:
    """
    Creates a fully-connected neural network model.

    :param layer_sizes: a list of the sizes of the layers including the input
        layer
    :param initialisation: the initialisation method to use for the weights of
        the layers
    :param activation: the activation function to use for the hidden and output
        layers
    :return: the fully-connected neural network model
    """
    if len(layer_sizes) < 2:
        raise ValueError

    if initialisation is None:
        initialisation = 'glorot_uniform'

    model = Sequential()
    model.add(InputLayer(input_shape=layer_sizes[0]))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(
            layer_size,
            kernel_initializer=initialisation,
            activation=activation))

    return model
