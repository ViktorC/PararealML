from __future__ import annotations

from typing import Optional, Tuple, Sequence, NamedTuple

import tensorflow as tf


class Loss(NamedTuple):
    """
    A collection of the various losses of a physics-informed DeepONet.
    """
    diff_eq_loss: tf.Tensor
    ic_loss: tf.Tensor
    bc_losses: Optional[Tuple[tf.Tensor, tf.Tensor]]
    weighted_total_loss: tf.Tensor

    def __str__(self):
        string = f'Weighted Total: {self.weighted_total_loss}; ' + \
                 f'DE: {self.diff_eq_loss}; ' + \
                 f'IC: {self.ic_loss}'

        if self.bc_losses:
            string += f'; Dirichlet BC: {self.bc_losses[0]}; ' + \
                      f'Neumann BC: {self.bc_losses[1]}'

        return string

    @classmethod
    def construct(
            cls,
            diff_eq_loss: tf.Tensor,
            ic_loss: tf.Tensor,
            bc_losses: Optional[Tuple[tf.Tensor, tf.Tensor]],
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float) -> Loss:
        """
        Calculates the weighted total loss given the weights for the different
        components of the total loss and returns a Loss instance.

        :param diff_eq_loss: the differential equation loss tensor
        :param ic_loss: the initial condition loss tensor
        :param bc_losses: a tuple of the Dirichlet and Neumann boundary
            condition loss tensors
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :return: the losses including the weighted total
        """
        weighted_total_loss = \
            tf.scalar_mul(diff_eq_loss_weight, diff_eq_loss) + \
            tf.scalar_mul(ic_loss_weight, ic_loss)
        if bc_losses:
            weighted_total_loss += \
                tf.scalar_mul(bc_loss_weight, bc_losses[0] + bc_losses[1])
        return Loss(diff_eq_loss, ic_loss, bc_losses, weighted_total_loss)

    @classmethod
    def mean(
            cls,
            losses: Sequence[Loss],
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float) -> Loss:
        """
        Computes the mean of the provided losses.

        :param losses: the losses to average over
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics-informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics-informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics-informed loss
        :return: the mean loss
        """
        diff_eq_losses = []
        ic_losses = []
        dirichlet_bc_losses = []
        neumann_bc_losses = []
        for loss in losses:
            diff_eq_losses.append(loss.diff_eq_loss)
            ic_losses.append(loss.ic_loss)
            if loss.bc_losses:
                dirichlet_bc_losses.append(loss.bc_losses[0])
                neumann_bc_losses.append(loss.bc_losses[1])

        mean_diff_eq_loss = tf.reduce_mean(tf.stack(diff_eq_losses), axis=0)
        mean_ic_loss = tf.reduce_mean(tf.stack(ic_losses), axis=0)
        mean_bc_losses = None \
            if len(dirichlet_bc_losses) + len(neumann_bc_losses) == 0 \
            else (tf.reduce_mean(tf.stack(dirichlet_bc_losses), axis=0),
                  tf.reduce_mean(tf.stack(neumann_bc_losses), axis=0))

        return cls.construct(
            mean_diff_eq_loss,
            mean_ic_loss,
            mean_bc_losses,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)
