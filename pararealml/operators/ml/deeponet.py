from typing import Sequence, Optional, Union, Tuple

import tensorflow as tf

from pararealml.operators.ml.fnn_regressor import FNNRegressor


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
            branch_input_size: int,
            trunk_input_size: int,
            latent_output_size: int,
            output_size: int,
            branch_hidden_layer_sizes: Optional[Sequence[int]] = None,
            trunk_hidden_layer_sizes: Optional[Sequence[int]] = None,
            combiner_hidden_layer_sizes: Optional[Sequence[int]] = None,
            branch_initialization: str = 'glorot_uniform',
            trunk_initialization: str = 'glorot_uniform',
            combiner_initialization: str = 'glorot_uniform',
            branch_activation: Optional[str] = 'tanh',
            trunk_activation: Optional[str] = 'tanh',
            combiner_activation: Optional[str] = 'tanh'):
        """
        :param branch_input_size: the size of the input layer of the branch net
        :param trunk_input_size: the size of the input layer of the trunk net
        :param latent_output_size: the size of the output layers of the branch
            and trunk nets
        :param output_size: the size of the output layer of the combiner net
        :param branch_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the branch net
        :param trunk_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the trunk net
        :param combiner_hidden_layer_sizes: a list of the sizes of the hidden
            layers of the combiner net
        :param branch_initialization: the initialization method to use for the
            weights of the branch net
        :param trunk_initialization: the initialization method to use for the
            weights of the trunk net
        :param combiner_initialization: the initialization method to use for
            the weights of the combiner net
        :param branch_activation: the activation function to use for the hidden
            layers of the branch net
        :param trunk_activation: the activation function to use for the hidden
            layers of the trunk net
        :param combiner_activation: the activation function to use for the
            hidden layers of the combiner net
        """
        if branch_input_size < 1 or trunk_input_size < 1 \
                or latent_output_size < 1 or output_size < 1:
            raise ValueError(
                'all input and output sizes must be greater than zero')

        super(DeepONet, self).__init__()

        self._branch_input_size = branch_input_size
        self._trunk_input_size = trunk_input_size
        self._latent_output_size = latent_output_size
        self._output_size = output_size

        if branch_hidden_layer_sizes is None:
            branch_hidden_layer_sizes = []
        if trunk_hidden_layer_sizes is None:
            trunk_hidden_layer_sizes = []
        if combiner_hidden_layer_sizes is None:
            combiner_hidden_layer_sizes = []

        self._branch_net = FNNRegressor(
            [branch_input_size] +
            list(branch_hidden_layer_sizes) +
            [latent_output_size],
            branch_initialization,
            branch_activation)
        self._trunk_net = FNNRegressor(
            [trunk_input_size] +
            list(trunk_hidden_layer_sizes) +
            [latent_output_size],
            trunk_initialization,
            trunk_activation)
        self._combiner_net = FNNRegressor(
            [3 * latent_output_size] +
            list(combiner_hidden_layer_sizes) +
            [output_size],
            combiner_initialization,
            combiner_activation
        )

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
        trunk_output = self._trunk_net.call(trunk_input)
        combiner_input = tf.concat(
            [branch_output, trunk_output, branch_output * trunk_output],
            axis=1
        )
        return self._combiner_net.call(combiner_input)
