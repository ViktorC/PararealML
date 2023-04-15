from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import VanDerPolEquation
from pararealml.initial_condition import ConstantInitialCondition
from pararealml.initial_value_problem import InitialValueProblem

diff_eq = VanDerPolEquation(0.25)
cp = ConstrainedProblem(diff_eq)
ic = ConstantInitialCondition(cp, [1.0, 0.0])
ivp = InitialValueProblem(cp, (0.0, 40.0), ic)
