import numpy as np

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.parareal import *
from pararealml.utils.time import mpi_time

diff_eq = DiffusionEquation(2)
mesh = Mesh([(0., 10.), (0., 10.)], [.5, .5])
bcs = [
    (
        DirichletBoundaryCondition(
            lambda x, t: np.full((len(x), 1), 1.5), is_static=True
        ),
        DirichletBoundaryCondition(
            lambda x, t: np.full((len(x), 1), 1.5), is_static=True
        )
    ),
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        )
    )
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    [
        (
            np.array([5., 5.]),
            np.array([
                [1., 0.],
                [0., 1.]
            ])
        )
    ],
    [1000.]
)
ivp = InitialValueProblem(cp, (0., 40.), ic)

f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .001)
g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .01)
g_g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
p_g = PararealOperator(g, g_g, 0., max_iterations=2)

p = PararealOperator(f, g, .01)
p_p = PararealOperator(f, p_g, .01)
p_g_g = PararealOperator(f, g_g, .01)

mpi_time('original_f_op')(f.solve)(ivp)
mpi_time('original_g_op')(g.solve)(ivp)
mpi_time('g_of_parareal_g_op')(g_g.solve)(ivp)
mpi_time('parareal_g_op')(p_g.solve)(ivp)
mpi_time('original_parareal_op')(p.solve)(ivp)
mpi_time('parareal_with_parareal_g_op')(p_p.solve)(ivp)
mpi_time('parareal_with_g_of_parareal_g_op')(p_g_g.solve)(ivp)
