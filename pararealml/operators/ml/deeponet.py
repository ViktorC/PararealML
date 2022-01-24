from typing import Sequence, Optional, Union, Tuple

import tensorflow as tf

from pararealml.operators.ml.fnn_regressor import FNNRegressor


class DeepONet(tf.keras.Model):
    """
    A Deep Operator Network model.

    See: https://arxiv.org/abs/1910.03193
    """

    def __init__(
            self,
            branch_layer_sizes: Sequence[int],
            trunk_layer_sizes: Sequence[int],
            output_size: int,
            branch_initialization: str = 'glorot_uniform',
            trunk_initialization: str = 'glorot_uniform',
            branch_activation: Optional[str] = 'tanh',
            trunk_activation: Optional[str] = 'tanh'):
        """
        :param branch_layer_sizes: a list of the sizes of the layers of the
            branch net; the last layer must match the last
        :param trunk_layer_sizes: a list of the sizes of the layers of the
            trunk net
        :param branch_initialization: the initialization method to use for the
            weights of the branch net
        :param trunk_initialization: the initialization method to use for the
            weights of the trunk net
        :param output_size: the number of columns in the model's rank-2 output
            tensor
        :param branch_activation: the activation function to use for the layers
            of the branch net
        :param trunk_activation: the activation function to use for the layers
            of the trunk net
        """
        if branch_layer_sizes[-1] != trunk_layer_sizes[-1]:
            raise ValueError(
                'last branch layer must be the same size as last trunk layer')
        if output_size <= 0 or branch_layer_sizes[-1] % output_size != 0:
            raise ValueError(
                'output size must be a divisor of final branch layer\'s size')

        super(DeepONet, self).__init__()

        self._branch_input_size = branch_layer_sizes[0]
        self._trunk_input_size = trunk_layer_sizes[0]
        self._output_size = output_size

        self._branch_net = FNNRegressor(
            branch_layer_sizes,
            branch_initialization,
            branch_activation)
        self._trunk_net = FNNRegressor(
            trunk_layer_sizes,
            trunk_initialization,
            trunk_activation)

    @tf.function
    def call(
            self,
            inputs:
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]]
    ) -> tf.Tensor:
        if isinstance(inputs, tuple):
            u = inputs[0]
            t = inputs[1]
            x = inputs[2]
            branch_input = u
            trunk_input = t if x is None else tf.concat([t, x], axis=1)
        else:
            branch_input = inputs[:, :self._branch_input_size]
            trunk_input = inputs[:, self._branch_input_size:]

        branch_output = self._branch_net.call(branch_input)
        branch_output = tf.reshape(
            branch_output,
            (tf.shape(branch_output)[0], -1, self._output_size))
        trunk_output = self._trunk_net.call(trunk_input)
        trunk_output = tf.reshape(
            trunk_output,
            (tf.shape(trunk_output)[0], -1, self._output_size))
        return tf.math.reduce_sum(branch_output * trunk_output, axis=1)
