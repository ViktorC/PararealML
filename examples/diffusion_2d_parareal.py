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
p = PararealOperator(f, g, .01)

mpi_time('fine')(f.solve)(ivp)
mpi_time('coarse')(g.solve)(ivp)
mpi_time('parareal')(p.solve)(ivp)
