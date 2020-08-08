from typing import Union, Callable, Tuple, Sequence, Any

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import all_estimators
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

SKLearnRegressionModel = Union[
    tuple([_class for name, _class in all_estimators()
           if issubclass(_class, RegressorMixin)])
]

RegressionModel = Union[
    SKLearnRegressionModel,
    KerasRegressor,
    GridSearchCV,
    RandomizedSearchCV,
    Pipeline
]


def train_regression_model(
        model: RegressionModel,
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = .2,
        score_func: Callable[[np.ndarray, np.ndarray], float] =
        mean_squared_error
) -> Tuple[float, float]:
    """
    Fits the regression model to the training share of the provided data points
    using random splitting and it returns the loss of the model evaluated on
    both the training and test data sets.

    :param model: the regression model to train
    :param x: the inputs
    :param y: the target outputs
    :param test_size: the fraction of all data points that should be used
        for testing
    :param score_func: the prediction scoring function to use
    :return: the training and test losses
    """
    assert 0. <= test_size < 1.
    train_size = 1. - test_size

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=train_size,
        test_size=test_size)

    model.fit(x_train, y_train)

    y_train_hat = model.predict(x_train)
    y_test_hat = model.predict(x_test)
    train_score = score_func(y_train, y_train_hat)
    test_score = score_func(y_test, y_test_hat)
    return train_score, test_score


def create_keras_regressor(
        layers: Sequence[Layer],
        optimiser: str = 'adam',
        loss: str = 'mse',
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: int = 0,
        **kwargs: Any,
) -> KerasRegressor:
    """
    Creates a Keras regression model.

    :param layers: the layers of the neural network
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
        model = Sequential(layers)
        model.compile(optimizer=optimiser, loss=loss)
        return model

    return KerasRegressor(
        build_fn=build_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        **kwargs)


def limit_visible_gpus():
    """
    If there are GPUs available, it sets the GPU corresponding to the MPI rank
    of the process as the only device visible to Tensorflow.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        comm = MPI.COMM_WORLD
        assert len(gpus) == comm.size
        tf.config.experimental.set_visible_devices(gpus[comm.rank], 'GPU')
