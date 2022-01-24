from typing import Union, Any

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


class SKLearnKerasRegressor(KerasRegressor):
    """
    A wrapper for Keras regression models to implement the implicit
    Scikit-learn model interface.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            optimizer: Union[str, Optimizer] = 'adam',
            loss: str = 'mse',
            epochs: int = 1000,
            batch_size: int = 64,
            verbose: bool = False,
            **kwargs: Any):
        """
        :param model: the Keras regression model
        :param optimizer: the optimizer to use
        :param loss: the loss function to use
        :param epochs: the number of training epochs
        :param batch_size: the training batch size
        :param verbose: whether training information should be printed to the
            stdout stream
        :param kwargs: additional parameters to the Keras regression model
        """
        def build_model() -> tf.keras.Model:
            model.compile(optimizer=optimizer, loss=loss)
            return model

        super(SKLearnKerasRegressor, self).__init__(
            build_fn=build_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs)
