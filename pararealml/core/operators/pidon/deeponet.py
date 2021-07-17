from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense


class DeepONet:
    """
    A DeepONet model.
    """

    def __init__(
            self,
            branch_layers: Sequence[int],
            trunk_layers: Sequence[int],
            activation: str = 'relu',
            initialisation: str = 'he_normal'
    ):
        """
        :param branch_layers: a list of the sizes of the input, hidden, and
            output layers of the branch net
        :param trunk_layers: a list of the sizes of the input, hidden, and
            output layers of the trunk net
        :param activation: the activation function to use for the hidden layers
        :param initialisation: the initialisation method to use for the weights
        """
        if branch_layers[-1] != trunk_layers[-1]:
            raise ValueError

        self._branch_net = \
            self._create_fnn(branch_layers, activation, initialisation)
        self._trunk_net = \
            self._create_fnn(trunk_layers, activation, initialisation)

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

    def init(
            self,
            branch_input_shape: Sequence[int],
            trunk_input_shape: Sequence[int]):
        """
        Initialises the model.

        :param branch_input_shape: the shape of the input of the branch net
        :param trunk_input_shape: the shape of the input of the trunk net
        """
        self._branch_net.build(branch_input_shape)
        self._trunk_net.build(trunk_input_shape)

    def predict(self, u: Tensor, x: Tensor) -> Tensor:
        """
        Predicts (y ∘ u)(x).

        :param u: sensor readings of the value of the function u at a set of
            points
        :param x: the input variables of the composed function
        :return: the predicted value of (y ∘ u)(x)
        """
        branch_prediction = self._branch_net.apply(u)
        trunk_prediction = self._trunk_net.apply(x)
        return tf.tensordot(branch_prediction, trunk_prediction, axes=1)

    @staticmethod
    def _create_fnn(
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
