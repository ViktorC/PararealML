import numpy as np
import pytest
from sympy import Symbol

from pararealml.differential_equation import DifferentialEquation, Symbols, \
    SymbolicEquationSystem, Lhs


def test_symbols_ode():
    symbols = Symbols(0, 3)
    assert symbols.t is not None
    assert symbols.y.shape == (3,)
    assert symbols.y_gradient is None
    assert symbols.y_hessian is None
    assert symbols.y_divergence is None
    assert symbols.y_curl is None
    assert symbols.y_laplacian is None


def test_symbols_pde():
    symbols = Symbols(3, 4)
    assert symbols.t is not None
    assert symbols.y.shape == (4,)
    assert symbols.y_gradient.shape == (4, 3)
    assert symbols.y_hessian.shape == (4, 3, 3)
    assert symbols.y_divergence.shape == (4, 4, 4)
    assert symbols.y_curl.shape == (4, 4, 4, 3)
    assert symbols.y_laplacian.shape == (4,)


def test_symbolic_equation_system_with_no_equations():
    with pytest.raises(ValueError):
        SymbolicEquationSystem([])


def test_symbolic_equation_system_with_mismatched_lhs_and_rhs_sizes():
    with pytest.raises(ValueError):
        SymbolicEquationSystem([2 * Symbol('p')], [Lhs.D_Y_OVER_D_T, Lhs.Y])


def test_symbolic_equation_system():
    rhs = [Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d'), Symbol('e')]
    lhs = [Lhs.D_Y_OVER_D_T, Lhs.Y, Lhs.D_Y_OVER_D_T, Lhs.Y_LAPLACIAN, Lhs.Y]
    eq_sys = SymbolicEquationSystem(rhs, lhs)

    assert np.array_equal(eq_sys.rhs, rhs)
    assert np.array_equal(eq_sys.lhs_types, lhs)

    expected_d_y_over_d_t_indices = [0, 2]
    expected_y_indices = [1, 4]
    expected_y_laplacian_indices = [3]
    assert np.array_equal(
        eq_sys.equation_indices_by_type(Lhs.D_Y_OVER_D_T),
        expected_d_y_over_d_t_indices)
    assert np.array_equal(
        eq_sys.equation_indices_by_type(Lhs.Y),
        expected_y_indices)
    assert np.array_equal(
        eq_sys.equation_indices_by_type(Lhs.Y_LAPLACIAN),
        expected_y_laplacian_indices)


def test_differential_equation_with_wrong_number_of_equations():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(2, 2)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem([
                self._symbols.y_laplacian[1]
            ])

    with pytest.raises(ValueError):
        TestDiffEq()


def test_differential_equation_with_invalid_symbol():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(0, 2)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem([
                Symbol('x') * self._symbols.y[0],
                self._symbols.t * self._symbols.y[1]
            ])

    with pytest.raises(ValueError):
        TestDiffEq()


def test_differential_equation_with_missing_d_y_over_d_t_lhs_in_pde_system():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(1, 2)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem(
                [
                    self._symbols.t * self._symbols.y[0],
                    self._symbols.y_gradient[0, 0]
                ], [Lhs.Y_LAPLACIAN, Lhs.Y])

    with pytest.raises(ValueError):
        TestDiffEq()


def test_differential_equation_with_non_d_y_over_d_t_lhs_in_ode_system():
    class TestDiffEq(DifferentialEquation):

        def __init__(self):
            super(TestDiffEq, self).__init__(0, 2)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem(
                [
                    self._symbols.t * self._symbols.y[0],
                    self._symbols.t * self._symbols.y[1]
                ], [Lhs.D_Y_OVER_D_T, Lhs.Y_LAPLACIAN])

    with pytest.raises(ValueError):
        TestDiffEq()
