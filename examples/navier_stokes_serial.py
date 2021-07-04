from pararealml import *

diff_eq = NavierStokesStreamFunctionVorticityEquation(5000.)
mesh = Mesh(((-2.5, 2.5), (0., 4.)), (.05, .05))
bcs = (
    (DirichletBoundaryCondition(lambda x, t: (1., .1), is_static=True),
     DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True)),
    (DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True),
     DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(cp, lambda x: (.0, .0))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .02)
solution = solver.solve(ivp)
solution.plot('navier_stokes', n_images=8)
