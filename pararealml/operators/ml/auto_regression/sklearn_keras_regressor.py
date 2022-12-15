from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf


class SKLearnKerasRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    A wrapper for Keras regression models to implement the implicit
    Scikit-learn model interface.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: Union[str, tf.optimizers.Optimizer] = "adam",
        loss: str = "mse",
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: bool = False,
        max_predict_batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        :param model: the Keras regression model
        :param optimizer: the optimizer to use
        :param loss: the loss function to use
        :param epochs: the number of training epochs
        :param batch_size: the training batch size
        :param verbose: whether training information should be printed to the
            stdout stream
        :param max_predict_batch_size: the maximum batch size to use for
            predictions
        :param kwargs: additional parameters to the Keras regression model
        """
        if max_predict_batch_size is not None and max_predict_batch_size < 1:
            raise ValueError(
                "the maximum prediction batch size "
                f"({max_predict_batch_size}) must be greater than 0"
            )

        self._max_predict_batch_size = max_predict_batch_size

        def build_model() -> tf.keras.Model:
            model.compile(optimizer=optimizer, loss=loss)
            return model

        super(SKLearnKerasRegressor, self).__init__(
            build_fn=build_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if self._max_predict_batch_size:
            kwargs["batch_size"] = self._max_predict_batch_size
        kwargs = self.filter_sk_params(tf.keras.Model.predict, kwargs)
        return self.model.predict(x, **kwargs)
