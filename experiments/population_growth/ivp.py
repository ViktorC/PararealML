from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import PopulationGrowthEquation
from pararealml.initial_condition import ConstantInitialCondition
from pararealml.initial_value_problem import InitialValueProblem

diff_eq = PopulationGrowthEquation(r=0.75)
cp = ConstrainedProblem(diff_eq)
ic = ConstantInitialCondition(cp, [1.0])
ivp = InitialValueProblem(cp, (0.0, 5.0), ic)
