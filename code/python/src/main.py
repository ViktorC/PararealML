from src.diff_eq import RabbitPopulationDiffEq
from src.runge_kutta import ForwardEulerMethod, RK4
from src.parareal import Parareal


diff_eq = RabbitPopulationDiffEq(10000., 5e-5)
fine_integrator = RK4()
coarse_integrator = ForwardEulerMethod()

fine_d_t = .025
coarse_d_t = 2 * fine_d_t
t_max = 100000.
k = 2

parareal = Parareal(diff_eq, fine_integrator, coarse_integrator, fine_d_t, coarse_d_t, t_max, k)
parareal.run()
