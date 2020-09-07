from pararealml import *

diff_eq = NavierStokes2DEquation(5000.)
mesh = UniformGrid(((-2.5, 2.5), (0., 4.)), (.05, .05))
bcs = (
    (DirichletBoundaryCondition(lambda x: (1., .1)),
     DirichletBoundaryCondition(lambda x: (.0, .0))),
    (DirichletBoundaryCondition(lambda x: (.0, .0)),
     DirichletBoundaryCondition(lambda x: (.0, .0)))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda x: (.0, .0))
ivp = InitialValueProblem(cp, (0., 100.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .02)
solution = solver.solve(ivp)
solution.plot('navier_stokes', n_images=50)
