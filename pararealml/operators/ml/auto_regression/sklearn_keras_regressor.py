from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf


class SKLearnKerasRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    A wrapper for Keras regression models to implement the implicit
    Scikit-learn model interface.
    """

    def __init__(
        self,
        build_fn: Callable[..., tf.keras.Model],
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: bool = False,
        max_predict_batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        :param build_fn: a function that compiles and returns the Keras model
            to wrap
        :param epochs: the number of training epochs
        :param batch_size: the training batch size
        :param verbose: whether training information should be printed to the
            stdout stream
        :param max_predict_batch_size: the maximum batch size to use for
            predictions
        :param kwargs: additional parameters to the Keras regression model
        """
        super(SKLearnKerasRegressor, self).__init__(
            build_fn=build_fn,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

        self.max_predict_batch_size = max_predict_batch_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        if (
            self.max_predict_batch_size is None
            or len(x) <= self.max_predict_batch_size
        ):
            return self._infer(tf.convert_to_tensor(x, tf.float32)).numpy()

        batch_start_ind = 0
        outputs = []
        while batch_start_ind < len(x):
            batch_end_ind = min(
                batch_start_ind + self.max_predict_batch_size, len(x)
            )
            batch = x[batch_start_ind:batch_end_ind]
            outputs.append(
                self._infer(tf.convert_to_tensor(batch, tf.float32)).numpy()
            )
            batch_start_ind += len(batch)

        return np.concatenate(outputs, axis=0)

    @tf.function
    def _infer(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Propagates the inputs through the underlying model.

        :param inputs: the model inputs
        :return: the model outputs
        """
        return self.model(inputs)
