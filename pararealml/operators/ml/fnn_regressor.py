from typing import Optional, Sequence

import tensorflow as tf


class FNNRegressor(tf.keras.Sequential):
    """
    A fully-connected feedforward neural network regression model.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        initialization: str = "glorot_uniform",
        activation: Optional[str] = "tanh",
    ):
        """
        :param layer_sizes: a list of the sizes of the layers including the
            input layer
        :param initialization: the initialization method to use for the weights
            of the layers
        :param activation: the activation function to use for the hidden layers
        """
        if len(layer_sizes) < 2:
            raise ValueError(
                f"number of layers ({len(layer_sizes)}) must be at least 2"
            )

        super(FNNRegressor, self).__init__()

        self.add(tf.keras.layers.InputLayer(input_shape=layer_sizes[0]))
        for layer_size in layer_sizes[1:-1]:
            self.add(
                tf.keras.layers.Dense(
                    layer_size,
                    kernel_initializer=initialization,
                    activation=activation,
                )
            )
        self.add(
            tf.keras.layers.Dense(
                layer_sizes[-1], kernel_initializer=initialization
            )
        )
