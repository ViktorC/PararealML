import pytest

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    LorenzEquation,
    PopulationGrowthEquation,
)
from pararealml.operators.ml.deeponet import DeepOSubNetArgs
from pararealml.operators.ml.pidon.pi_deeponet import PIDeepONet


def test_pi_deeponet_with_zero_latent_size():
    cp = ConstrainedProblem(PopulationGrowthEquation())

    with pytest.raises(ValueError):
        PIDeepONet(
            cp, 0, DeepOSubNetArgs(), DeepOSubNetArgs(), DeepOSubNetArgs()
        )


def test_pi_deeponet_with_wrong_loss_weight_length():
    cp = ConstrainedProblem(LorenzEquation())

    with pytest.raises(ValueError):
        PIDeepONet(
            cp,
            3,
            DeepOSubNetArgs(),
            DeepOSubNetArgs(),
            DeepOSubNetArgs(),
            diff_eq_loss_weight=[0.0, 1.0],
        )


def test_pi_deeponet_loss_weight_broadcasting():
    cp = ConstrainedProblem(LorenzEquation())
    pidon = PIDeepONet(
        cp,
        3,
        DeepOSubNetArgs(),
        DeepOSubNetArgs(),
        DeepOSubNetArgs(),
        diff_eq_loss_weight=2.0,
    )

    assert pidon.differential_equation_loss_weights == (2.0, 2.0, 2.0)
    assert pidon.initial_condition_loss_weights == (1.0, 1.0, 1.0)
    assert pidon.boundary_condition_loss_weights == (1.0, 1.0, 1.0)
