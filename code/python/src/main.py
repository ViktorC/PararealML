from mpi4py import MPI
from sklearn.linear_model import LinearRegression

from src.diff_eq import RabbitPopulationDiffEq
from src.integrator import ForwardEulerMethod, RK4
from src.operator import ConventionalOperator, MLOperator
from src.parareal import Parareal

diff_eq = RabbitPopulationDiffEq(1000., 1e-4, 0., 50000.)
f = ConventionalOperator(RK4(), .01)
g = ConventionalOperator(ForwardEulerMethod(), .1)
k = 1

solver = Parareal(f, g, k)
solver.solve(diff_eq)

ml_g = MLOperator(LinearRegression(), g, 100., 10)
ml_g.train_model(diff_eq)

ml_solver = Parareal(f, ml_g, k)
ml_solver.solve(diff_eq)

if MPI.COMM_WORLD.rank == 0:
    start_time = MPI.Wtime()
    y_x_max = f.trace(
        diff_eq, diff_eq.y_0(), diff_eq.x_0(), diff_eq.x_max())[-1]
    end_time = MPI.Wtime()

    print(f'Serial fine solution: {y_x_max}')
    print(f'Serial execution took {end_time - start_time}s')
