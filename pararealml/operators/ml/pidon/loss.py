from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import tensorflow as tf


class Loss(NamedTuple):
    """
    A collection of the various losses of a physics-informed DeepONet.
    """

    diff_eq_loss: tf.Tensor
    ic_loss: tf.Tensor
    bc_losses: Optional[Tuple[tf.Tensor, tf.Tensor]]
    model_loss: Optional[tf.Tensor]
    weighted_total_loss: tf.Tensor

    def __str__(self):
        string = (
            f"Weighted Total: {self.weighted_total_loss}; "
            + f"DE: {self.diff_eq_loss}; "
            + f"IC: {self.ic_loss}"
        )
        if self.bc_losses:
            string += (
                f"; Dirichlet BC: {self.bc_losses[0]}; "
                + f"Neumann BC: {self.bc_losses[1]}"
            )
        if self.model_loss is not None:
            string += f"; Model: {self.model_loss}"
        return string

    @classmethod
    @tf.function
    def construct(
        cls,
        diff_eq_loss: tf.Tensor,
        ic_loss: tf.Tensor,
        bc_losses: Optional[Tuple[tf.Tensor, tf.Tensor]],
        model_loss: Optional[tf.Tensor],
        diff_eq_loss_weights: Sequence[float],
        ic_loss_weights: Sequence[float],
        bc_loss_weights: Sequence[float],
    ) -> Loss:
        """
        Calculates the weighted total loss given the weights for the different
        components of the total loss and returns a Loss instance.

        :param diff_eq_loss: the differential equation loss tensor
        :param ic_loss: the initial condition loss tensor
        :param bc_losses: a tuple of the Dirichlet and Neumann boundary
            condition loss tensors
        :param diff_eq_loss_weights: the weights of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weights: the weights of the initial condition part of
            the total physics-informed loss
        :param bc_loss_weights: the weights of the boundary condition part of
            the total physics-informed loss
        :param model_loss: model loss such as regularization penalty
        :return: the losses including the weighted total
        """
        weighted_total_loss = tf.multiply(
            tf.constant(diff_eq_loss_weights), diff_eq_loss
        ) + tf.multiply(tf.constant(ic_loss_weights), ic_loss)
        if bc_losses:
            weighted_total_loss += tf.multiply(
                tf.constant(bc_loss_weights), bc_losses[0] + bc_losses[1]
            )
        if model_loss is not None:
            weighted_total_loss += model_loss
        return Loss(
            diff_eq_loss, ic_loss, bc_losses, model_loss, weighted_total_loss
        )

    @classmethod
    @tf.function
    def mean(
        cls,
        losses: Sequence[Loss],
        diff_eq_loss_weights: Sequence[float],
        ic_loss_weights: Sequence[float],
        bc_loss_weights: Sequence[float],
    ) -> Loss:
        """
        Computes the mean of the provided losses.

        :param losses: the losses to average over
        :param diff_eq_loss_weights: the weights of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weights: the weights of the initial condition part of
            the total physics-informed loss
        :param bc_loss_weights: the weights of the boundary condition part of
            the total physics-informed loss
        :return: the mean loss
        """
        diff_eq_losses = []
        ic_losses = []
        dirichlet_bc_losses = []
        neumann_bc_losses = []
        model_losses = []
        for loss in losses:
            diff_eq_losses.append(loss.diff_eq_loss)
            ic_losses.append(loss.ic_loss)
            if loss.bc_losses:
                dirichlet_bc_losses.append(loss.bc_losses[0])
                neumann_bc_losses.append(loss.bc_losses[1])
            if loss.model_loss is not None:
                model_losses.append(loss.model_loss)

        mean_diff_eq_loss = tf.reduce_mean(tf.stack(diff_eq_losses), axis=0)
        mean_ic_loss = tf.reduce_mean(tf.stack(ic_losses), axis=0)
        mean_bc_losses = (
            None
            if len(dirichlet_bc_losses) + len(neumann_bc_losses) == 0
            else (
                tf.reduce_mean(tf.stack(dirichlet_bc_losses), axis=0),
                tf.reduce_mean(tf.stack(neumann_bc_losses), axis=0),
            )
        )
        mean_model_loss = (
            tf.reduce_mean(tf.stack(model_losses), axis=0)
            if model_losses
            else None
        )

        return cls.construct(
            mean_diff_eq_loss,
            mean_ic_loss,
            mean_bc_losses,
            mean_model_loss,
            diff_eq_loss_weights,
            ic_loss_weights,
            bc_loss_weights,
        )
