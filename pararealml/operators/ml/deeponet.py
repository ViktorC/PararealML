from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple, Union

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
        branch_net_args: DeepOSubNetArgs,
        trunk_net_args: DeepOSubNetArgs,
        combiner_net_args: DeepOSubNetArgs,
    ):
        """
        :param branch_input_size: the size of the input layer of the branch net
        :param trunk_input_size: the size of the input layer of the trunk net
        :param latent_output_size: the size of the output layers of the branch
            and trunk nets
        :param output_size: the size of the output layer of the combiner net
        :param branch_net_args: the arguments for the branch net
        :param trunk_net_args: the arguments for the trunk net
        :param combiner_net_args: the arguments for the combiner net
        """
        if (
            branch_input_size < 1
            or trunk_input_size < 1
            or latent_output_size < 1
            or output_size < 1
        ):
            raise ValueError(
                "all input and output sizes must be greater than zero"
            )

        super(DeepONet, self).__init__()

        self._branch_input_size = branch_input_size
        self._trunk_input_size = trunk_input_size
        self._latent_output_size = latent_output_size
        self._output_size = output_size

        self._branch_net = FNNRegressor(
            [branch_input_size]
            + list(branch_net_args.hidden_layer_sizes)
            + [latent_output_size],
            branch_net_args.initialization,
            branch_net_args.activation,
        )
        self._trunk_net = FNNRegressor(
            [trunk_input_size]
            + list(trunk_net_args.hidden_layer_sizes)
            + [latent_output_size],
            trunk_net_args.initialization,
            trunk_net_args.activation,
        )
        self._combiner_net = FNNRegressor(
            [3 * latent_output_size]
            + list(combiner_net_args.hidden_layer_sizes)
            + [output_size],
            combiner_net_args.initialization,
            combiner_net_args.activation,
        )

    @property
    def branch_net(self) -> FNNRegressor:
        """
        The model's branch net that processes the initial condition sensor
        readings.
        """
        return self._branch_net

    @property
    def trunk_net(self) -> FNNRegressor:
        """
        The model's trunk net that processes the domain coordinates.
        """
        return self._trunk_net

    @property
    def combiner_net(self) -> FNNRegressor:
        """
        The model's combiner net that combines the outputs of the branch and
        trunk nets.
        """
        return self._combiner_net

    @tf.function
    def get_trainable_parameters(self) -> tf.Tensor:
        """
        All the trainable parameters of the model flattened into a single-row
        matrix.
        """
        return tf.concat(
            [tf.reshape(var, (1, -1)) for var in self.trainable_variables],
            axis=1,
        )

    @tf.function
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

    @tf.function
    def call(
        self,
        inputs: Union[
            tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
        ],
    ) -> tf.Tensor:
        if isinstance(inputs, tuple):
            u = inputs[0]
            t = inputs[1]
            x = inputs[2]
            branch_input = u
            trunk_input = t if x is None else tf.concat([t, x], axis=1)
        else:
            branch_input = inputs[:, : self._branch_input_size]
            trunk_input = inputs[:, self._branch_input_size :]

        branch_output = self._branch_net.call(branch_input)
        trunk_output = self._trunk_net.call(trunk_input)
        combiner_input = tf.concat(
            [branch_output, trunk_output, branch_output * trunk_output], axis=1
        )
        return self._combiner_net.call(combiner_input)


class DeepOSubNetArgs(NamedTuple):
    """
    Arguments for a DeepONet sub-network including a list of the sizes of the
    hidden layers, the initialization method to use for the model's weights,
    and the activation function to apply to the hidden layers.
    """

    hidden_layer_sizes: Sequence[int] = []
    initialization: str = "glorot_uniform"
    activation: Optional[str] = "tanh"
