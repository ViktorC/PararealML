import numpy as np

from pararealml import *
from pararealml.operators.ode import *

g = 6.6743e-11

minute = 60
hour = 60 * minute
day = 24 * hour

m_earth = 5.97e24
m_moon = 7.34767e22
d_earth_moon = 3.844e8
orbital_v_moon = np.sqrt(g * m_earth / d_earth_moon)

v_x = 5.
v_y = 5.
v_length = np.sqrt(v_x ** 2 + v_y ** 2)

masses = [m_earth, m_moon, m_moon, m_moon, m_moon]
positions = [
    0., 0.,
    d_earth_moon, 0.,
    0., d_earth_moon,
    -d_earth_moon, 0.,
    0., -d_earth_moon
]
velocities = [
    0., 0.,
    -v_x / v_length * orbital_v_moon, v_y / v_length * orbital_v_moon,
    -v_y / v_length * orbital_v_moon, -v_x / v_length * orbital_v_moon,
    v_x / v_length * orbital_v_moon, -v_y / v_length * orbital_v_moon,
    v_y / v_length * orbital_v_moon, v_x / v_length * orbital_v_moon
]

diff_eq = NBodyGravitationalEquation(2, masses)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(
    cp,
    lambda _: np.array(positions + velocities)
)
ivp = InitialValueProblem(cp, (0., 120 * day), ic)

solver = ODEOperator('DOP853', minute)
solution = solver.solve(ivp)

for plot in solution.generate_plots():
    plot.show().close()
