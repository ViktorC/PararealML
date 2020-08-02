import numpy as np
from fipy import LinearCGSSolver
from sklearn.ensemble import RandomForestRegressor

from src.experiment import Experiment
from src.core.boundary_condition import NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import CahnHilliardEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, FDMOperator, \
    SolutionRegressionOperator, PINNOperator, OperatorRegressionOperator

diff_eq = CahnHilliardEquation(2, 1., .01)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.1, .1))
bvp = BoundaryValueProblem(
    diff_eq,
    mesh,
    ((NeumannCondition(lambda x: (0., 0.)),
      NeumannCondition(lambda x: (0., 0.))),
     (NeumannCondition(lambda x: (0., 0.)),
      NeumannCondition(lambda x: (0., 0.)))))
ic = DiscreteInitialCondition(
    bvp,
    .05 * np.random.uniform(-1., 1., bvp.y_shape(False)),
    False)
ivp = InitialValueProblem(
    bvp,
    (0., 2.),
    ic)

f = FVMOperator(LinearCGSSolver(), .01)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g_pinn = PINNOperator(.5, f.vertex_oriented)
g_sol_reg = SolutionRegressionOperator(.5, f.vertex_oriented)
g_op_reg = OperatorRegressionOperator(.5, f.vertex_oriented)

threshold = .1

experiment = Experiment(ivp, f, g, g_pinn, g_sol_reg, g_op_reg, threshold)

experiment.solve_serial_fine()
experiment.solve_serial_coarse()
experiment.solve_parallel()

experiment.train_coarse_pinn(
    (50,) * 4, 'tanh', 'Glorot normal',
    n_domain=2000,
    n_initial=200,
    n_boundary=100,
    n_test=400,
    n_epochs=5000,
    optimiser='adam',
    learning_rate=.001)
experiment.solve_serial_coarse_pinn()
experiment.solve_parallel_pinn()

experiment.train_coarse_sol_reg(
    RandomForestRegressor(),
    subsampling_factor=.01)
experiment.solve_serial_coarse_sol_reg()
experiment.solve_parallel_sol_reg()

experiment.train_coarse_op_reg(
    RandomForestRegressor(),
    iterations=10,
    noise_sd=1e-5)
experiment.solve_serial_coarse_op_reg()
experiment.solve_parallel_op_reg()
