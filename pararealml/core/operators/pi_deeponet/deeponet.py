from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Sequential

from pararealml.utils.ml import create_fnn


class DeepONet:
    """
    A DeepONet model.
    """

    def __init__(
            self,
            branch_layers: Sequence[int],
            trunk_layers: Sequence[int],
            activation: str,
            initialisation: str
    ):
        if branch_layers[-1] != trunk_layers[-1]:
            raise ValueError

        self._branch_net = \
            create_fnn(branch_layers, activation, initialisation)
        self._trunk_net = create_fnn(trunk_layers, activation, initialisation)

    @property
    def branch_net(self) -> Optional[Sequential]:
        return self._branch_net

    @property
    def trunk_net(self) -> Optional[Sequential]:
        return self._trunk_net

    def init(
            self,
            branch_input_shape: Sequence[int],
            trunk_input_shape: Sequence[int]):
        self._branch_net.build(branch_input_shape)
        self._trunk_net.build(trunk_input_shape)

    def predict(self, u: Tensor, x: Tensor) -> Tensor:
        branch_prediction = self._branch_net.apply(u)
        trunk_prediction = self._trunk_net.apply(x)
        return tf.tensordot(branch_prediction, trunk_prediction, axes=1)
