from pararealml.boundary_condition import ConstantFluxBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import BurgersEquation
from pararealml.initial_condition import MarginalBetaProductInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh

diff_eq = BurgersEquation(2, re=25)
mesh = Mesh([(0.0, 1.0), (0.0, 1.0)], [0.05, 0.05])
bcs = [
    (
        ConstantFluxBoundaryCondition([0.0, 0.0]),
        ConstantFluxBoundaryCondition([0.0, 0.0]),
    ),
    (
        ConstantFluxBoundaryCondition([0.0, 0.0]),
        ConstantFluxBoundaryCondition([0.0, 0.0]),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0.0, 0.5)
ic = MarginalBetaProductInitialCondition(
    cp, [[(7, 5), (6, 6)], [(6, 6), (3, 9)]]
)
ivp = InitialValueProblem(cp, t_interval, ic)
