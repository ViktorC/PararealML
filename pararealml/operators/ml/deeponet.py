from __future__ import annotations

from typing import Optional

import tensorflow as tf


class DeepONet(tf.keras.Model):
    """
    A Deep Operator Network model.

    It differs from the referenced DeepONet architecture in that it includes a
    combiner network that takes the outputs of the branch and trunk nets in
    addition to the product of these outputs to produce the final output of
    the model.

    See: https://arxiv.org/abs/1910.03193
    """

    def __init__(
        self,
        branch_net: tf.keras.Model,
        trunk_net: tf.keras.Model,
        combiner_net: tf.keras.Model,
        branch_net_input_size: Optional[int] = None,
    ):
        """
        :param branch_net: the model's branch net that processes the initial
            condition sensor readings
        :param trunk_net: the model's trunk net that processes the domain
            coordinates
        :param combiner_net: the model's combiner net that combines the outputs
            of the branch and trunk nets
        :param branch_net_input_size: the size of the branch net's input
        """
        super(DeepONet, self).__init__()
        self._branch_net = branch_net
        self._trunk_net = trunk_net
        self._combiner_net = combiner_net
        self._branch_net_input_size = branch_net_input_size

    @property
    def branch_net(self) -> tf.keras.Model:
        """
        The model's branch net that processes the initial condition sensor
        readings.
        """
        return self._branch_net

    @property
    def trunk_net(self) -> tf.keras.Model:
        """
        The model's trunk net that processes the domain coordinates.
        """
        return self._trunk_net

    @property
    def combiner_net(self) -> tf.keras.Model:
        """
        The model's combiner net that combines the outputs of the branch and
        trunk nets.
        """
        return self._combiner_net

    @property
    def branch_net_input_size(self) -> Optional[int]:
        """
        The input size of the branch net if one is explicitly provided.
        """
        return self._branch_net_input_size

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        branch_net_input_size = (
            self._branch_net.layers[0].input_shape[1]
            if self._branch_net_input_size is None
            else self._branch_net_input_size
        )
        branch_input = inputs[:, :branch_net_input_size, ...]
        trunk_input = inputs[:, branch_net_input_size:, ...]

        branch_output = self._branch_net(
            branch_input, training=training, mask=mask
        )
        trunk_output = self._trunk_net(
            trunk_input, training=training, mask=mask
        )
        combiner_input = tf.concat(
            [branch_output, trunk_output, branch_output * trunk_output], axis=1
        )
        return self._combiner_net(combiner_input, training=training, mask=mask)
