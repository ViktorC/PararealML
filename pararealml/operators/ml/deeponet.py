from __future__ import annotations

from typing import Optional, Tuple, Union

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
    ):
        """
        :param branch_net: the model's branch net that processes the initial
            condition sensor readings
        :param trunk_net: the model's trunk net that processes the domain
            coordinates
        :param combiner_net: the model's combiner net that combines the outputs
            of the branch and trunk nets
        """
        super(DeepONet, self).__init__()
        self._branch_net = branch_net
        self._trunk_net = trunk_net
        self._combiner_net = combiner_net

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

    def get_trainable_parameters(self) -> tf.Tensor:
        """
        All the trainable parameters of the model flattened into a single-row
        matrix.
        """
        return tf.concat(
            [tf.reshape(var, (1, -1)) for var in self.trainable_variables],
            axis=1,
        )

    def set_trainable_parameters(self, value: tf.Tensor):
        """
        Sets the trainable parameters of the model to the values provided.

        :param value: the parameters values flattened into a single-row matrix
        """
        offset = 0
        for var in self.trainable_variables:
            var_size = tf.reduce_prod(var.shape)
            var.assign(
                tf.reshape(value[0, offset : offset + var_size], var.shape)
            )
            offset += var_size

    def call(
        self,
        inputs: Union[
            tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
        ],
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if isinstance(inputs, tuple):
            u = inputs[0]
            t = inputs[1]
            x = inputs[2]
            branch_input = u
            trunk_input = t if x is None else tf.concat([t, x], axis=1)
        else:
            branch_net_input_size = self._branch_net.layers[0].input_shape[1]
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
