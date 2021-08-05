from typing import Optional, Sequence, Dict, Any, Union, Tuple, NamedTuple, \
    List

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs
from pararealml.core.operators.pidon.auto_differentiator import \
    AutoDifferentiator
from pararealml.core.operators.pidon.data_set import DataSet
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
            iterations: int,
            optimizer: Union[str, Dict[str, Any]],
            training_data: DataSet,
            training_domain_batch_size: int,
            training_boundary_batch_size: int = 0,
            test_data: Optional[DataSet] = None,
            test_domain_batch_size: int = 0,
            test_boundary_batch_size: int = 0,
            iterations_before_test: int = 10,
            diff_eq_loss_weight: float = 1,
            ic_loss_weight: float = 1,
            bc_loss_weight: float = 1,
            verbose: bool = False
    ) -> Tuple[Sequence[LossTensors], Sequence[LossTensors]]:
        """
        Trains the branch and trunk net parameters by minimising the physics
        informed loss function over various initial conditions.

        :param iterations: the number of training batches/iterations
        :param optimizer: the optimizer to use to minimize the loss function
        :param training_data: the data set providing the training mini batches
        :param training_domain_batch_size: the training domain data batch size
        :param training_boundary_batch_size: the training boundary data batch
            size
        :param test_data: the data set providing the test mini batches
        :param test_domain_batch_size: the test domain data batch size
        :param test_boundary_batch_size: the test boundary data batch size
        :param iterations_before_test: the number of training iterations to
            perform before performing a test iteration
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
            between tests
        :param verbose: whether loss information should be periodically printed
            to the console
        :return: the training and test loss histories
        """
        if iterations < 1:
            raise ValueError
        if training_domain_batch_size <= 0 or training_boundary_batch_size < 0:
            raise ValueError
        if test_domain_batch_size < 0 or test_boundary_batch_size < 0:
            raise ValueError
        if test_data and test_domain_batch_size <= 0:
            raise ValueError
        if iterations_before_test < 1:
            raise ValueError

        optimizer_instance = tf.keras.optimizers.get(optimizer)

        training_loss_history = []
        test_loss_history = []
        for i in range(iterations):
            with tf.GradientTape(persistent=True) as tape:
                self._differentiator.tape = tape

                training_losses = self._total_loss(
                        training_data,
                        training_domain_batch_size,
                        training_boundary_batch_size,
                        diff_eq_loss_weight,
                        ic_loss_weight,
                        bc_loss_weight)
                training_loss_history.append(training_losses)

                if test_data and \
                        (i == 0 or (i + 1) % iterations_before_test == 0):
                    test_losses = self._total_loss(
                            test_data,
                            test_domain_batch_size,
                            test_boundary_batch_size,
                            diff_eq_loss_weight,
                            ic_loss_weight,
                            bc_loss_weight)
                    test_loss_history.append(test_losses)

                    if verbose:
                        print('Iteration: ', i)
                        print('DE Loss:',
                              'Training -', training_losses.diff_eq_loss,
                              'Test -', test_losses.diff_eq_loss)
                        print('IC Loss:',
                              'Training -', training_losses.ic_loss,
                              'Test -', test_losses.ic_loss)
                        print('Dirichlet BC Loss:',
                              'Training -', training_losses.dirichlet_bc_loss,
                              'Test -', test_losses.dirichlet_bc_loss)
                        print('Neumann BC Loss:',
                              'Training -', training_losses.neumann_bc_loss,
                              'Test -', test_losses.neumann_bc_loss)
                        print('Total Loss:',
                              'Training -',
                              training_losses.total_weighted_loss,
                              'Test -', test_losses.total_weighted_loss)

            optimizer_instance.minimize(
                training_losses.total_weighted_loss,
                self._branch_net.trainable_variables +
                self._trunk_net.trainable_variables,
                tape=tape)

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

    def _total_loss(
            self,
            data_set: DataSet,
            domain_batch_size: int,
            boundary_batch_size: Optional[int],
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float
    ) -> LossTensors:
        """
        Computes and returns the weighted total loss, the differential equation
        loss, the initial condition loss, the Dirichlet boundary condition
        loss, and the Neumann boundary condition loss.

        :param data_set: the data set to sample the batches from
        :param domain_batch_size: the domain data batch size
        :param boundary_batch_size: the boundary data batch size
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
        :return: the weighted total loss and the mean squared differential
            equation, initial condition, Dirichlet boundary condition, and
            Neumann boundary condition losses
        """
        diff_eq_loss, ic_loss = self._domain_loss(data_set, domain_batch_size)
        dirichlet_bc_loss, neumann_bc_loss = \
            self._boundary_loss(data_set, boundary_batch_size)

        total_weighted_loss = \
            tf.math.scalar_mul(
                diff_eq_loss_weight,
                diff_eq_loss) + \
            tf.math.scalar_mul(
                ic_loss_weight,
                ic_loss) + \
            tf.math.scalar_mul(
                bc_loss_weight,
                dirichlet_bc_loss + neumann_bc_loss)

        return LossTensors(
            total_weighted_loss,
            diff_eq_loss,
            ic_loss,
            dirichlet_bc_loss,
            neumann_bc_loss)

    def _domain_loss(
            self,
            data_set: DataSet,
            batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the differential equation loss and the initial
        condition loss over the domain of the IVP.

        :param data_set: the data set to sample a domain data batch from
        :param batch_size: the size of the batch to sample
        :return: the mean squared differential equation and initial condition
            errors
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension

        domain_batch = data_set.get_domain_batch(batch_size)
        domain_u_tensor = tf.convert_to_tensor(
            data_set.get_initial_condition_batch(batch_size), dtype=tf.float32)
        domain_t_tensor = tf.convert_to_tensor(
            domain_batch.t, dtype=tf.float32)

        if x_dimension:
            domain_x_tensor = tf.convert_to_tensor(
                domain_batch.x, dtype=tf.float32)
            sensor_x_tensor = tf.convert_to_tensor(
                self._sensor_points, dtype=tf.float32)
            sensor_t_tensor = tf.zeros((sensor_x_tensor.shape[0], 1))
        else:
            domain_x_tensor = None
            sensor_x_tensor = None
            sensor_t_tensor = tf.zeros((1, 1))

        diff_eq_loss = self._mean_squared_differential_equation_error(
            domain_u_tensor, domain_t_tensor, domain_x_tensor)
        ic_loss = self._mean_squared_initial_condition_error(
            domain_u_tensor, sensor_t_tensor, sensor_x_tensor)
        return diff_eq_loss, ic_loss

    def _boundary_loss(
            self,
            data_set: DataSet,
            batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns the boundary condition loss.

        :param data_set: the data set to sample a boundary data batch from
        :param batch_size: the size of the batch to sample
        :return: the mean squared Dirichlet and Neumann boundary condition
            errors
        """
        diff_eq = self._cp.differential_equation
        if not diff_eq.x_dimension or batch_size < 1:
            return tf.zeros((diff_eq.y_dimension,)), \
                tf.zeros((diff_eq.y_dimension,))

        boundary_batch = data_set.get_boundary_batch(batch_size)
        boundary_u_tensor = tf.convert_to_tensor(
            data_set.get_initial_condition_batch(batch_size), dtype=tf.float32)
        boundary_t_tensor = tf.convert_to_tensor(
            boundary_batch.collocation_points.t, dtype=tf.float32)
        boundary_x_tensor = tf.convert_to_tensor(
            boundary_batch.collocation_points.x, dtype=tf.float32)
        boundary_y_tensor = tf.convert_to_tensor(
            boundary_batch.y, dtype=tf.float32)
        boundary_d_y_over_d_n_tensor = tf.convert_to_tensor(
            boundary_batch.d_y_over_d_n, dtype=tf.float32)
        boundary_axes_tensor = tf.convert_to_tensor(
            boundary_batch.axes, dtype=tf.float32)

        return self._mean_squared_boundary_condition_errors(
                boundary_u_tensor,
                boundary_t_tensor,
                boundary_x_tensor,
                boundary_y_tensor,
                boundary_d_y_over_d_n_tensor,
                boundary_axes_tensor)

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
