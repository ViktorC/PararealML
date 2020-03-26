from mpi4py import MPI
from sklearn.linear_model import LinearRegression

from src.diff_eq import RabbitPopulationDiffEq
from src.integrator import ExplicitMidpointMethod, RK4
from src.operator import ConventionalOperator, MLOperator
from src.parareal import Parareal


def time_parallel_solver_and_print_result(parallel_solver, solver_name):
    comm.barrier()
    start_time = MPI.Wtime()
    y_max = parallel_solver.solve(diff_eq, threshold)[-1]
    comm.barrier()
    end_time = MPI.Wtime()
    if comm.rank == 0:
        print(f'{solver_name} solution: {y_max}; '
              f'execution took {end_time - start_time}s')


def time_operator_and_print_result(operator, operator_name):
    start_time = MPI.Wtime()
    y_max = operator.trace(
        diff_eq, diff_eq.y_0(), diff_eq.x_0(), diff_eq.x_max())[-1]
    end_time = MPI.Wtime()
    print(f'{operator_name} solution: {y_max}; '
          f'execution took {end_time - start_time}s')


comm = MPI.COMM_WORLD

diff_eq = RabbitPopulationDiffEq(1000., 1e-4, 0., 50000.)

f = ConventionalOperator(RK4(), .01)
g = ConventionalOperator(ExplicitMidpointMethod(), .25)
g_ml = MLOperator(LinearRegression(), g, 100., 10)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = 1e-3

time_parallel_solver_and_print_result(parareal_ml, 'Parareal ML w/ training')
time_parallel_solver_and_print_result(parareal_ml, 'Parareal ML w/o training')
time_parallel_solver_and_print_result(parareal, 'Parareal')

if comm.rank == 0:
    time_operator_and_print_result(g, 'Coarse')
    time_operator_and_print_result(g_ml, 'Coarse ML')
    time_operator_and_print_result(f, 'Fine')
    print(f'Analytic solution: {diff_eq.exact_y(diff_eq.x_max())}')
