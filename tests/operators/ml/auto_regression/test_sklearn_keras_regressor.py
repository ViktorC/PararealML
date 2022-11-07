import pytest

from pararealml.operators.ml.auto_regression import SKLearnKerasRegressor
from pararealml.utils.tf import create_fnn_regressor


def test_sklearn_keras_regressor_with_zero_max_predict_batch_size():
    with pytest.raises(ValueError):
        SKLearnKerasRegressor(
            create_fnn_regressor([7, 50, 50, 1]), max_predict_batch_size=0
        )
