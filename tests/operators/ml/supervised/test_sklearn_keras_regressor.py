import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

from pararealml.operators.ml.supervised import SKLearnKerasRegressor
from pararealml.utils.rand import set_random_seed

FEATURES = np.random.rand(100, 5)
TARGET = ((FEATURES + 2) ** 2).sum(
    axis=1, keepdims=True
) + 0.1 * np.random.rand(100, 1)

TRAIN_FEATURES = FEATURES[:80]
TRAIN_TARGET = TARGET[:80]

TEST_FEATURES = FEATURES[80:]
TEST_TARGET = TARGET[80:]


def build_model(
    hidden_layer_size: int, optimizer: str, loss: str
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(hidden_layer_size, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss=loss)
    return model


def test_sklearn_keras_regressor_without_lazy_loading():
    regressor = SKLearnKerasRegressor(
        build_model,
        epochs=250,
        batch_size=20,
        max_predict_batch_size=10,
        validation_split=0.1,
        hidden_layer_size=100,
        optimizer="adam",
        loss="mse",
    )
    regressor.fit(TRAIN_FEATURES, TRAIN_TARGET)

    train_score = regressor.score(TRAIN_FEATURES, TRAIN_TARGET)
    train_prediction = regressor.predict(TRAIN_FEATURES)
    assert np.isclose(
        -((TRAIN_TARGET - train_prediction) ** 2).mean(), train_score
    )

    test_score = regressor.score(TEST_FEATURES, TEST_TARGET)
    test_prediction = regressor.predict(TEST_FEATURES)
    assert np.isclose(
        -((TEST_TARGET - test_prediction) ** 2).mean(), test_score
    )


def test_sklearn_keras_regressor_with_lazy_loading():
    regressor_with_lazy_loading = SKLearnKerasRegressor(
        build_model,
        epochs=250,
        batch_size=20,
        max_predict_batch_size=10,
        validation_split=0.1,
        lazy_load_to_gpu=True,
        prefetch_buffer_size=1,
        hidden_layer_size=100,
        optimizer="adam",
        loss="mse",
    )
    regressor_with_lazy_loading.fit(TRAIN_FEATURES, TRAIN_TARGET)

    regressor_without_lazy_loading = SKLearnKerasRegressor(
        lambda _: regressor_with_lazy_loading.model
    )
    regressor_without_lazy_loading.model = regressor_with_lazy_loading.model

    assert np.isclose(
        regressor_with_lazy_loading.score(TRAIN_FEATURES, TRAIN_TARGET),
        regressor_without_lazy_loading.score(TRAIN_FEATURES, TRAIN_TARGET),
    )


def test_sklearn_keras_regressor_with_hyperparameter_tuning():
    set_random_seed(0)

    search = GridSearchCV(
        SKLearnKerasRegressor(build_model, verbose=0),
        {
            "hidden_layer_size": [5, 10, 20],
            "optimizer": ["adam"],
            "loss": ["mse"],
            "epochs": [50, 100, 200],
        },
        cv=5,
        verbose=5,
    )
    search.fit(TRAIN_FEATURES, TRAIN_TARGET)

    assert search.best_params_["hidden_layer_size"] == 20
    assert search.best_params_["epochs"] == 200
