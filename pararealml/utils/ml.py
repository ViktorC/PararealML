from typing import Any

import tensorflow as tf
from mpi4py import MPI
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


def limit_visible_gpus():
    """
    If there are GPUs available, it sets the GPU corresponding to the MPI rank
    of the process as the only device visible to Tensorflow.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        comm = MPI.COMM_WORLD
        if len(gpus) != comm.size:
            raise ValueError
        tf.config.experimental.set_visible_devices(gpus[comm.rank], 'GPU')


def create_keras_regressor(
        model: Sequential,
        optimiser: str = 'adam',
        loss: str = 'mse',
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: int = 0,
        **kwargs: Any,
) -> KerasRegressor:
    """
    Creates a Keras regression model.

    :param model: the neural network
    :param optimiser: the optimiser to use
    :param loss: the loss function to use
    :param epochs: the number of training epochs
    :param batch_size: the training batch size
    :param verbose: whether training information should be printed to the
        stdout stream
    :param kwargs: additional parameters to the Keras regression model
    :return: the regression model
    """
    def build_model():
        model.compile(optimizer=optimiser, loss=loss)
        return model

    return KerasRegressor(
        build_fn=build_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        **kwargs)
