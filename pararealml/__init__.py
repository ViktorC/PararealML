from __future__ import absolute_import

from pararealml.boundary_condition import (
    BoundaryCondition,
    CauchyBoundaryCondition,
    ConstantBoundaryCondition,
    ConstantFluxBoundaryCondition,
    ConstantValueBoundaryCondition,
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
    VectorizedBoundaryConditionFunction,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.constraint import Constraint, apply_constraints_along_last_axis
from pararealml.differential_equation import (
    LHS,
    BurgerEquation,
    CahnHilliardEquation,
    ConvectionDiffusionEquation,
    DifferentialEquation,
    DiffusionEquation,
    LorenzEquation,
    LotkaVolterraEquation,
    NavierStokesEquation,
    NBodyGravitationalEquation,
    PopulationGrowthEquation,
    ShallowWaterEquation,
    SIREquation,
    SymbolicEquationSystem,
    Symbols,
    VanDerPolEquation,
    WaveEquation,
)
from pararealml.initial_condition import (
    ConstantInitialCondition,
    ContinuousInitialCondition,
    DiscreteInitialCondition,
    GaussianInitialCondition,
    InitialCondition,
    MarginalBetaProductInitialCondition,
    VectorizedInitialConditionFunction,
    vectorize_ic_function,
)
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import (
    CoordinateSystem,
    Mesh,
    from_cartesian_coordinates,
    to_cartesian_coordinates,
    unit_vectors_at,
)
from pararealml.plot import (
    AnimatedPlot,
    ContourPlot,
    NBodyPlot,
    PhaseSpacePlot,
    Plot,
    QuiverPlot,
    ScatterPlot,
    SpaceLinePlot,
    StreamPlot,
    SurfacePlot,
    TimePlot,
)
from pararealml.solution import Solution

__all__ = [
    "BoundaryCondition",
    "DirichletBoundaryCondition",
    "NeumannBoundaryCondition",
    "CauchyBoundaryCondition",
    "ConstantBoundaryCondition",
    "ConstantValueBoundaryCondition",
    "ConstantFluxBoundaryCondition",
    "VectorizedBoundaryConditionFunction",
    "vectorize_bc_function",
    "ConstrainedProblem",
    "apply_constraints_along_last_axis",
    "Constraint",
    "Symbols",
    "LHS",
    "SymbolicEquationSystem",
    "DifferentialEquation",
    "PopulationGrowthEquation",
    "LotkaVolterraEquation",
    "LorenzEquation",
    "SIREquation",
    "VanDerPolEquation",
    "NBodyGravitationalEquation",
    "DiffusionEquation",
    "ConvectionDiffusionEquation",
    "WaveEquation",
    "CahnHilliardEquation",
    "BurgerEquation",
    "ShallowWaterEquation",
    "NavierStokesEquation",
    "InitialCondition",
    "DiscreteInitialCondition",
    "ConstantInitialCondition",
    "ContinuousInitialCondition",
    "GaussianInitialCondition",
    "MarginalBetaProductInitialCondition",
    "VectorizedInitialConditionFunction",
    "vectorize_ic_function",
    "InitialValueProblem",
    "CoordinateSystem",
    "Mesh",
    "to_cartesian_coordinates",
    "from_cartesian_coordinates",
    "unit_vectors_at",
    "Plot",
    "AnimatedPlot",
    "TimePlot",
    "PhaseSpacePlot",
    "NBodyPlot",
    "SpaceLinePlot",
    "ContourPlot",
    "SurfacePlot",
    "ScatterPlot",
    "StreamPlot",
    "QuiverPlot",
    "Solution",
]
