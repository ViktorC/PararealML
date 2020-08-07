from typing import Union, Callable, Tuple, Sequence, Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    train_test_split
from sklearn.utils import all_estimators
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

SKLearnRegressionModel = Union[
    tuple([_class for name, _class in all_estimators()
           if issubclass(_class, RegressorMixin)])
]
RegressionModel = Union[SKLearnRegressionModel, KerasRegressor]
SearchCV = Union[GridSearchCV, RandomizedSearchCV]


def train_regression_model(
        model: Union[RegressionModel, SearchCV],
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = .2,
        score_func: Callable[[np.ndarray, np.ndarray], float] =
        mean_squared_error
) -> Tuple[RegressionModel, float, float]:
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
    :return: the fitted model, the training loss, and the test loss
    """
    assert 0. <= test_size < 1.
    train_size = 1. - test_size

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=train_size,
        test_size=test_size)

    model.fit(x_train, y_train)

    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        model = model.best_estimator_

    y_train_hat = model.predict(x_train)
    y_test_hat = model.predict(x_test)
    train_score = score_func(y_train, y_train_hat)
    test_score = score_func(y_test, y_test_hat)
    return model, train_score, test_score


def create_keras_regressor(
        layers: Sequence[Layer],
        optimiser: str = 'adam',
        loss: str = 'mse',
        **kwargs: Any,
) -> KerasRegressor:
    """
    Creates a Keras regression model.

    :param layers: the layers of the neural network
    :param optimiser: the optimiser to use
    :param loss: the loss function to use
    :param kwargs: additional parameters to the Keras regression model
    :return: the regression model
    """
    def build_model():
        model = Sequential(layers)
        model.compile(optimizer=optimiser, loss=loss)
        return model

    return KerasRegressor(build_fn=build_model, **kwargs)
