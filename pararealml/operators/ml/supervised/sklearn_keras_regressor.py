from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CPU_DEVICE_TYPE = "CPU"


class SKLearnKerasRegressor:
    """
    A wrapper for Keras regression models to implement the implicit
    Scikit-learn model interface.
    """

    def __init__(
        self,
        build_fn: Callable[..., tf.keras.Model],
        batch_size: int = 256,
        epochs: int = 1000,
        verbose: Union[int, str] = "auto",
        callbacks: Sequence[tf.keras.callbacks.Callback] = (),
        validation_split: float = 0.0,
        validation_frequency: int = 1,
        lazy_load_to_gpu: bool = False,
        prefetch_buffer_size: int = 1,
        max_predict_batch_size: Optional[int] = None,
        **build_args: Any,
    ):
        """
        :param build_fn: a function that compiles and returns the Keras model
            to wrap
        :param batch_size: the training batch size
        :param epochs: the number of training epochs
        :param verbose: controls the level of training and evaluation
            information printed to the stdout stream
        :param callbacks: any callbacks for the training of the model
        :param validation_split: the proportion of the training data to use for
            validation
        :param validation_frequency: the number of training epochs between each
            validation
        :param lazy_load_to_gpu: whether to avoid loading the entire training
            data set onto the GPU all at once by using lazy loading instead
        :param prefetch_buffer_size: the number of batches to prefetch if using
            lazy loading to the GPU
        :param max_predict_batch_size: the maximum batch size to use for
            predictions
        :param build_args: all the parameters to pass to `build_fn`
        """
        self.build_fn = build_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_frequency = validation_frequency
        self.lazy_load_to_gpu = lazy_load_to_gpu
        self.prefetch_buffer_size = prefetch_buffer_size
        self.max_predict_batch_size = max_predict_batch_size
        self.build_args = build_args

        self._model: Optional[tf.keras.Model] = None

    @property
    def model(self) -> tf.keras.Model:
        """
        The underlying Tensorflow model.
        """
        return self._model

    @model.setter
    def model(self, model: tf.keras.Model):
        self._model = model

    def get_params(self, **_: Any) -> Dict[str, Any]:
        params = {
            "build_fn": self.build_fn,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
            "validation_split": self.validation_split,
            "lazy_load_to_gpu": self.lazy_load_to_gpu,
            "prefetch_buffer_size": self.prefetch_buffer_size,
            "max_predict_batch_size": self.max_predict_batch_size,
        }
        params.update(self.build_args)
        return params

    def set_params(self, **parameters: Any) -> SKLearnKerasRegressor:
        build_fn_arg_names = list(
            inspect.signature(self.build_fn).parameters.keys()
        )
        build_args = {}
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in build_fn_arg_names:
                build_args[key] = value
            else:
                raise ValueError(f"invalid parameter '{key}'")

        self.build_args.update(build_args)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> SKLearnKerasRegressor:
        self._model = self.build_fn(**self.build_args)

        if self.lazy_load_to_gpu:
            with tf.device(CPU_DEVICE_TYPE):
                if self.validation_split:
                    (
                        x_train,
                        x_validate,
                        y_train,
                        y_validate,
                    ) = train_test_split(
                        x,
                        y,
                        test_size=self.validation_split,
                    )
                    training_dataset = (
                        tf.data.Dataset.from_tensor_slices((x_train, y_train))
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )
                    validation_dataset = (
                        tf.data.Dataset.from_tensor_slices(
                            (x_validate, y_validate)
                        )
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )

                else:
                    training_dataset = (
                        tf.data.Dataset.from_tensor_slices((x, y))
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )
                    validation_dataset = None

            self._model.fit(
                training_dataset,
                epochs=self.epochs,
                validation_data=validation_dataset,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )

        else:
            self._model.fit(
                x,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                validation_freq=self.validation_frequency,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )

        return self

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

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.lazy_load_to_gpu:
            with tf.device(CPU_DEVICE_TYPE):
                dataset = (
                    tf.data.Dataset.from_tensor_slices((x, y))
                    .batch(self.batch_size)
                    .prefetch(self.prefetch_buffer_size)
                )

            loss = self._model.evaluate(dataset, verbose=self.verbose)
        else:
            loss = self._model.evaluate(x, y, verbose=self.verbose)

        if isinstance(loss, Sequence):
            return -loss[0]
        return -loss

    @tf.function
    def _infer(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Propagates the inputs through the underlying model.

        :param inputs: the model inputs
        :return: the model outputs
        """
        return self._model(inputs)
