from pararealml import *

diff_eq = NBodyGravitationalEquation(3, [5e10, 5e12, 5e10])
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: (-10., 0., 5.) + (0., 0., 0.) + (10., 0., -5.) +
              (0., .25, 0.) + (0., 5., 0.) + (0., -.25, .0))
ivp = InitialValueProblem(cp, (0., 10.), ic)

solver = ODEOperator('DOP853', 1e-4)
solution = solver.solve(ivp)
solution.plot('fine', n_images=20, smallest_marker_size=20)
