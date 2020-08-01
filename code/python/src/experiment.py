from typing import Union, Optional, Tuple

from deepxde.maps import FNN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator, RegressionOperator, PINNOperator, \
    RegressionModel
from src.core.parareal import PararealOperator
from src.core.solution import Solution
from src.utils.plot import plot
from src.utils.profile import profile
from src.utils.time import time


class Experiment:
    """
    A class representing a Parareal experiment using ML accelerated coarse
    operators.
    """

    def __init__(
            self,
            ivp: InitialValueProblem,
            f: Operator,
            g: Operator,
            g_reg: RegressionOperator,
            g_pinn: PINNOperator,
            tol: float):
        """
        :param ivp: the initial value problem to solve
        :param f: the fine operator
        :param g: the coarse operator
        :param g_reg: the regression model based coarse operator
        :param g_pinn: the PINN based coarse operator
        :param tol: the convergence tolerance of the Parareal framework
        """
        assert ivp is not None
        assert f is not None
        assert g is not None
        assert g_reg is not None
        assert g_pinn is not None
        assert tol > 0.

        self._ivp = ivp
        self._f = f
        self._g = g
        self._g_reg = g_reg
        self._g_pinn = g_pinn

        self._parareal = PararealOperator(f, g, tol)
        self._parareal_reg = PararealOperator(f, g_reg, tol)
        self._parareal_pinn = PararealOperator(f, g_pinn, tol)

    @profile
    @time
    def train_coarse_reg(
            self,
            model: Union[RegressionModel, GridSearchCV, RandomizedSearchCV],
            test_size: float = .2,
            subsampling_factor: Optional[float] = None):
        """
        Trains the regression model based coarse operator.

        :param model: the regression model
        :param test_size: the fraction of all data points that should be used
        for testing
        :param subsampling_factor: the fraction of all data points that should
        be sampled for training; it has to be greater than 0 and less than or
        equal to 1; if it is None, all data points will be used
        """
        self._g_reg.train(
            self._ivp,
            self._g,
            model,
            test_size=test_size,
            subsampling_factor=subsampling_factor)

    @profile
    @time
    def train_coarse_pinn(
            self,
            hidden_layer_sizes: Tuple[int, ...],
            activation_function: str,
            initialisation: str,
            **training_config: Union[int, float, str]):
        """
        :param hidden_layer_sizes: a tuple of ints representing the sizes of
        the hidden layers
        :param activation_function: the activation function to use
        :param initialisation: the initialisation to use
        :param training_config: the training configuration
        """
        diff_eq = self._ivp.boundary_value_problem.differential_equation
        x_dim = diff_eq.x_dimension + 1
        y_dim = diff_eq.y_dimension
        self._g_pinn.train(
            self._ivp,
            FNN(
                (x_dim,) + hidden_layer_sizes + (y_dim,),
                activation_function,
                initialisation),
            **training_config)

    @plot
    @time
    def solve_serial_fine(self) -> Solution:
        """
        Solves the IVP serially using the fine operator.
        """
        return self._f.solve(self._ivp)

    @plot
    @time
    def solve_serial_coarse(self) -> Solution:
        """
        Solves the IVP serially using the coarse operator.
        """
        return self._g.solve(self._ivp)

    @plot
    @time
    def solve_serial_coarse_reg(self) -> Solution:
        """
        Solves the IVP serially using the regression model based coarse
        operator.
        """
        return self._g_reg.solve(self._ivp)

    @plot
    @time
    def solve_serial_coarse_pinn(self) -> Solution:
        """
        Solves the IVP serially using the PINN based coarse operator.
        """
        return self._g_pinn.solve(self._ivp)

    @plot
    @time
    def solve_parallel(self) -> Solution:
        """
        Solves the IVP using the Parareal framework on top of the fine operator
        and the coarse operator
        """
        return self._parareal.solve(self._ivp)

    @plot
    @time
    def solve_parallel_reg(self) -> Solution:
        """
        Solves the IVP using the Parareal framework on top of the fine operator
        and the regression model based coarse operator
        """
        return self._parareal_reg.solve(self._ivp)

    @plot
    @time
    def solve_parallel_pinn(self) -> Solution:
        """
        Solves the IVP using the Parareal framework on top of the fine operator
        and the PINN based coarse operator
        """
        return self._parareal_pinn.solve(self._ivp)
