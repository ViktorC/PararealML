# PararealML

A Python library providing a Parareal framework based on a unified interface for various differential equation solvers including a finite volume method solver, its own finite difference method solver, and a range of other machine learning solvers.

## Main components

The main components of the library are the following.

### Differential Equation

All differential equations extend the `DifferentialEquation` base class. The unknown variables of all differential equations are vector-valued to support systems of differential equations. Both ordinary and partial differential equations are supported.

The list of currently supported differential equations is as follows:

 * population growth
 * Lotka-Volterra
 * Lorenz
 * gravitational n-body (2 or 3D, not supported by the `PINNOperator`)
 * diffusion (n-D)
 * wave (n-D)
 * Cahn-Hilliard (n-D)
 * Navier-Stokes (2 or 3D, not supported by the `FVMOperator` and the `PINNOperator`)

### Mesh

All meshes extend the `Mesh` base class. Meshes define the spatial domain of initial boundary value problems and the discretisation of this domain.

The library currently only supports uniform grids (of any dimensions for the `FDMOperator` and for up to 3 dimensions for the `FVMOperator` and the `PINNOperator`) but can be easily extended to support more complex meshes.

### Boundary Conditions

All boundary conditions extend the `BoundaryCondition` base class. Boundary conditions are functions of the spatial coordinates only.

The list of supported boundary conditions is:

 * Dirichlet
 * Neumann
 * Cauchy

### Constrained Problem

Constrained problems encapsulate either simple ordinary differential equations or partial differential equations coupled with a mesh and boundary conditions. This offers a level of abstraction over the two kinds of differential equations. Constrained problems are represented by the `ConstrainedProblem` class whose instances take a differential equation and optionally a mesh and a set of boundary conditions.

### Initial Conditions

All initial conditions extend the `InitialCondition` base class.

The supported initial condition types are:

 * discrete (defined by a NumPy array and whether it is vertex or cell-center oriented)
 * continuous (defined by a function)
    * Gaussian hump

### Initial Value Problem

Initial value problems are constrained problems associated with a time domain and initial conditions. They are represented by the `InitialValueProblem` class whose instances take a constrained problem, a tuple of two `float`s defining the time interval, and an initial condition instance.

### Operators

All operators extend the `Operator` base class. Operators are standalone differential equation solvers that can be used as operators in the Parareal framework as well. They are generally constructed by giving them a time step size. They then can solve initial value problems and return the trajectories of the solutions with a granularity defined by their time step sizes. The Parareal framework is an operator itself as well.

The list of supported operators is:

 * ODE operator (based on `SciPy`'s `solve_ivp`, can solve all ordinary differential equations)
 * FVM operator (finite volume method operator based on `FiPy`, can solve all partial differential equations except for Navier-Stokes)
 * FDM operator (a self-implemented fast finite difference method operator, can solve any differential equation)
 * machine learning operator
    * stateless machine learning operator
        * PINN operator (a physics-informed neural network operator based on `DeepXDE`, can solve all differential equations except for n-body and Navier-Stokes)
        * stateless regression operator (a regression operator trained on data generated by another operator, can solve any differential equation the trainer operator can)
    * stateful machine learning operator
        * stateful regression operator (a regression operator trained on data generated by another operator taking the previous value of the solution into account, can solve any differential equation the trainer operator can)
 * Parareal operator (can solve any differential equation its operators can)

## Examples

The [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder contains a range of different examples of using the library for solving various differential equations both in serial and parallel. The scripts also include examples of using machine learning operators. Furthermore, all the scripts used for our experiments and the generation of our figures for the dissertation can be found in this folder. These scripts use hardcoded seeds (defined in [rand.py](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/blob/master/code/python/src/utils/rand.py)) thus all our results are fully reproducible. 

## Getting started
To use the full feature set of the library, FiPy must be made available. The recommended way of doing that is installing it through Anaconda (see the [installation guide](https://www.ctcms.nist.gov/fipy/INSTALLATION.html)).

To use multiprocessing, the library also requires a working MPI implementation.

To install any other requirements of the library, run `make install`.

To perform linting, execute `make lint`. The library uses type-hints throughout. For `mypy` type checking, use the command `make type-check` (note that this may return a list of warnings including mostly false positives).

To run the unit tests, execute `make test`.

To run any of the examples from the [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder using 4 MPI processes, run `make run example={name of example file without extension}` (e.g. `make run example=diffusion_1d_serial`).

The code base is also thoroughly commented. To generate the Sphinx documentation from the docstrings, type `cd docs` and run `make html`. This generates the HTML documentation in the `./docs/_build/html` directory. To browse the documentation, just open `index.html`.