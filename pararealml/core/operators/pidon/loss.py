from __future__ import annotations

from typing import Optional, Tuple, Sequence

import tensorflow as tf


class Loss:
    """
    A collection of the various losses of a physics-informed DeepONet in
    tensor form.
    """

    def __init__(
            self,
            diff_eq_loss: tf.Tensor,
            ic_loss: tf.Tensor,
            bc_losses: Optional[Tuple[tf.Tensor, tf.Tensor]],
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float):
        """
        :param diff_eq_loss: the differential equation loss tensor
        :param ic_loss: the initial condition loss tensor
        :param bc_losses: a tuple of the Dirichlet and Neumann boundary
            condition loss tensors
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
        """
        self._diff_eq_loss = diff_eq_loss
        self._ic_loss = ic_loss
        self._bc_losses = bc_losses

        self._total_weighted_loss = \
            tf.scalar_mul(diff_eq_loss_weight, diff_eq_loss) + \
            tf.scalar_mul(ic_loss_weight, ic_loss)
        if bc_losses:
            self._total_weighted_loss += \
                tf.scalar_mul(bc_loss_weight, bc_losses[0] + bc_losses[1])

    def __str__(self):
        string = f'DE: {self._diff_eq_loss.numpy()}; ' + \
            f'IC: {self._ic_loss.numpy()}'

        if self._bc_losses:
            string += f'; Dirichlet BC: {self._bc_losses[0].numpy()}; ' + \
                f'Neumann BC: {self._bc_losses[1].numpy()}'

        return string

    @property
    def diff_eq_loss(self) -> tf.Tensor:
        """
        The differential equation loss tensor.
        """
        return self._diff_eq_loss

    @property
    def ic_loss(self) -> tf.Tensor:
        """
        The initial condition loss tensor.
        """
        return self._ic_loss

    @property
    def bc_losses(self) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
        """
        The Dirichlet and Neumann boundary condition loss tensors.
        """
        return self._bc_losses

    @property
    def total_weighted_loss(self) -> tf.Tensor:
        """
        The total weighted loss tensor.
        """
        return self._total_weighted_loss

    @staticmethod
    def mean(
            losses: Sequence[Loss],
            diff_eq_loss_weight: float,
            ic_loss_weight: float,
            bc_loss_weight: float) -> Loss:
        """
        Computes the mean of the provided losses.

        :param losses: the losses to average over
        :param diff_eq_loss_weight: the weight of the differential equation
            part of the total physics informed loss
        :param ic_loss_weight: the weight of the initial condition part of the
            total physics informed loss
        :param bc_loss_weight: the weight of the boundary condition part of the
            total physics informed loss
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

        return Loss(
            mean_diff_eq_loss,
            mean_ic_loss,
            mean_bc_losses,
            diff_eq_loss_weight,
            ic_loss_weight,
            bc_loss_weight)
