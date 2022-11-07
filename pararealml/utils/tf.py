import os
from typing import Optional, Sequence

import tensorflow as tf
from mpi4py import MPI


def use_cpu():
    """
    Ensures that Tensorflow does not use any GPUs.
    """
    tf.config.experimental.set_visible_devices([], "GPU")


def limit_visible_gpus():
    """
    If there are GPUs available, it sets the GPU corresponding to the MPI rank
    of the process as the only device visible to Tensorflow.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        comm = MPI.COMM_WORLD
        if len(gpus) != comm.size:
            raise ValueError(
                f"number of GPUs ({len(gpus)}) must match default "
                f"communicator size ({comm.size})"
            )
        tf.config.experimental.set_visible_devices(gpus[comm.rank], "GPU")


def use_deterministic_ops():
    """
    Ensures Tensorflow operations are deterministic.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def create_fnn_regressor(
    layer_sizes: Sequence[int],
    initialization: str = "glorot_uniform",
    hidden_layer_activation: Optional[str] = "tanh",
) -> tf.keras.Sequential:
    """
    Creates a fully-connected feedforward neural network regression model.

    :param layer_sizes: a list of the sizes of the layers including the
        input layer
    :param initialization: the initialization method to use for the weights
        of the layers
    :param hidden_layer_activation: the activation function to use for the
        hidden layers
    :return a fully-connected feedforward neural network regression model
    """
    if len(layer_sizes) < 2:
        raise ValueError(
            f"number of layers ({len(layer_sizes)}) must be at least 2"
        )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=layer_sizes[0]))
    for layer_size in layer_sizes[1:-1]:
        model.add(
            tf.keras.layers.Dense(
                layer_size,
                kernel_initializer=initialization,
                activation=hidden_layer_activation,
            )
        )
    model.add(
        tf.keras.layers.Dense(
            layer_sizes[-1], kernel_initializer=initialization
        )
    )
    return model
