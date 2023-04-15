import numpy as np

from pararealml.boundary_condition import (
    ConstantFluxBoundaryCondition,
    ConstantValueBoundaryCondition,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import DiffusionEquation
from pararealml.initial_condition import ContinuousInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import CoordinateSystem, Mesh


def ic_function(x: np.ndarray) -> np.ndarray:
    return np.cos((x[:, 0] - 2.0) * np.pi / 4.0).reshape((len(x), 1))


diff_eq = DiffusionEquation(2, 20.0)
mesh = Mesh(
    [(2.0, 6.0), (0.0, 2 * np.pi)], [0.2, np.pi / 10.0], CoordinateSystem.POLAR
)
bcs = [
    (
        ConstantValueBoundaryCondition([1.0]),
        ConstantFluxBoundaryCondition([0.0]),
    ),
    (
        ConstantFluxBoundaryCondition([0.0]),
        ConstantFluxBoundaryCondition([0.0]),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0.0, 2.0)
ic = ContinuousInitialCondition(cp, ic_function)
ivp = InitialValueProblem(cp, t_interval, ic)
