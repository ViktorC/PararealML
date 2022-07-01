import pytest

from pararealml.operators.ml.auto_regression import SKLearnKerasRegressor
from pararealml.operators.ml.fnn_regressor import FNNRegressor


def test_sklearn_keras_regressor_with_zero_max_predict_batch_size():
    with pytest.raises(ValueError):
        SKLearnKerasRegressor(
            FNNRegressor([7, 50, 50, 1]), max_predict_batch_size=0
        )
