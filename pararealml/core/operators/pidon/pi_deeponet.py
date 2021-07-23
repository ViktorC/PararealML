from typing import Optional, Sequence, Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense

from pararealml.core.initial_value_problem import ConstrainedProblem
from pararealml.core.differential_equation import Lhs
from pararealml.core.operators.pidon.collocation_point_sampler import \
    CollocationPointSet
from pararealml.core.operators.pidon.differentiation import gradient, laplacian
from pararealml.core.operators.pidon.pidon_symbol_mapper import PIDONSymbolMapper


class PIDeepONet:
    """
    A Physics Informed DeepONet model.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            n_sensor_points: int,
            latent_output_size: int,
            branch_net_hidden_layer_sizes: Optional[Sequence[int]] = None,
            trunk_net_hidden_layer_sizes: Optional[Sequence[int]] = None,
            activation: str = 'relu',
            initialisation: str = 'he_normal'
    ):
        """
        :param cp: the constrained problem to build a physics informed neural
            network around
        :param n_sensor_points: the number of sensor points to use to represent
            the input function of the DeepONet
        :param latent_output_size: the size of the latent output of the
            DeepONet model
        :param branch_net_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the branch net
        :param trunk_net_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the trunk net
        :param activation: the activation function to use for the hidden layers
        :param initialisation: the initialisation method to use for the weights
        """
        if branch_net_hidden_layer_sizes is None:
            branch_net_hidden_layer_sizes = []
        if trunk_net_hidden_layer_sizes is None:
            trunk_net_hidden_layer_sizes = []

        diff_eq = cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        self._cp = cp
        self._n_sensor_points = n_sensor_points
        self._latent_output_size = latent_output_size
        self._branch_net = create_fnn(
            [n_sensor_points] +
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

        symbol_mapper = PIDONSymbolMapper(cp)
        self._diff_eq_rhs_function, self._diff_eq_rhs_arg_funcs = \
            symbol_mapper.create_rhs_lambda_and_arg_functions()
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()

    @property
    def branch_net(self) -> Optional[Sequential]:
        """
        The branch net of the model.
        """
        return self._branch_net

    @property
    def trunk_net(self) -> Optional[Sequential]:
        """
        The trunk net of the model.
        """
        return self._trunk_net

    @tf.function
    def predict(self, u: Tensor, x: Tensor) -> Tensor:
        """
        Predicts (y ∘ u)(x).

        :param u: sensor readings of the value of the function u at a set of
            points
        :param x: the input variables of the composed function
        :return: the predicted value of (y ∘ u)(x)
        """
        y_dimension = self._cp.differential_equation.y_dimension
        branch_out = self._branch_net.apply(u).reshape(
            (-1, y_dimension, self._latent_output_size))
        trunk_out = self._trunk_net.apply(x).reshape(
            (-1, y_dimension, self._latent_output_size))
        combined_out = branch_out * trunk_out
        return tf.math.reduce_sum(combined_out, axis=-1)

    @tf.function
    def diff_eq_error(self, x: Tensor, y: Tensor) -> Tensor:
        """

        :param x:
        :param y:
        :return:
        """
        diff_eq = self._cp.differential_equation
        rhs = self._diff_eq_rhs_function(
            [func(x, y) for func in self._diff_eq_rhs_arg_funcs]
        )
        return tf.stack([
            self._diff_eq_lhs_functions[i](x, y) - rhs[i]
            for i in range(diff_eq.y_dimension)
        ], axis=-1)

    def train(self, collocation_points: CollocationPointSet, epochs: int):
        for i in range(epochs):
            ...

    def _create_diff_eq_lhs_functions(self) \
            -> Sequence[Callable[[Tensor, Tensor], Tensor]]:
        """
        Creates a sequence of functions of x and y representing the left hand
        side of the differential equation system.
        """
        diff_eq = self._cp.differential_equation
        lhs_functions = []
        for y_ind, lhs_type in \
                enumerate(diff_eq.symbolic_equation_system.lhs_types):
            if lhs_type == Lhs.D_Y_OVER_D_T:
                lhs_functions.append(
                    lambda x, y, _y_ind=y_ind:
                    gradient(x[:, -1:], y[:, _y_ind:_y_ind + 1], 0))
            elif lhs_type == Lhs.Y:
                lhs_functions.append(
                    lambda x, y, _y_ind=y_ind: y[:, _y_ind:_y_ind + 1])
            elif lhs_type == Lhs.Y_LAPLACIAN:
                lhs_functions.append(
                    lambda x, y, _y_ind=y_ind:
                    laplacian(
                        x[:, :-1],
                        y[:, _y_ind:_y_ind + 1],
                        self._cp.mesh.coordinate_system_type))
            else:
                raise ValueError

        return lhs_functions

    def _create_boundary_loss_functions(
            self,
            boundary_points: Sequence[Tuple[np.ndarray, np.ndarray]]):
        boundary_conditions = self._cp.boundary_conditions
        if len(boundary_conditions) != len(boundary_points):
            raise ValueError

        dirichlet_x = []
        dirichlet_y = []

        all_neumann_x = []
        all_neumann_d_y = []

        for i, bc_pair in enumerate(boundary_conditions):
            neumann_x = []
            neumann_d_y = []

            axis_boundary_points = boundary_points[i]
            for j in range(2):
                boundary_points = axis_boundary_points[j]
                bc = bc_pair[j]

                for k in range(boundary_points.shape[0]):
                    x = boundary_points[j]
                    if bc.has_y_condition:
                        y = bc.y_condition(x[1:], x[0])
                        if any([y_elem is not None for y_elem in y]):
                            dirichlet_x.append(x)
                            dirichlet_y.append(y)
                    if bc.has_d_y_condition:
                        d_y = bc.d_y_condition(x[1:], x[0])
                        if any([d_y_elem is not None for d_y_elem in d_y]):
                            neumann_x.append(x)
                            neumann_d_y.append(d_y)

            all_neumann_x.append(np.array(neumann_x))
            all_neumann_d_y.append(np.array(neumann_d_y))

        dirichlet_x = np.array(dirichlet_x)
        dirichlet_y = np.array(dirichlet_y)


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
    model.add(Input(shape=layer_sizes[0]))
    for layer_size in layer_sizes[1:-1]:
        model.add(Dense(
            layer_size,
            activation=activation,
            kernel_initializer=initialisation))
    model.add(Dense(layer_sizes[-1], kernel_initializer=initialisation))
    return model
