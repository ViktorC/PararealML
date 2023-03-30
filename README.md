# PararealML

[![Build Status](https://github.com/ViktorC/PararealML/actions/workflows/build.yml/badge.svg)](https://github.com/ViktorC/PararealML/actions/workflows/build.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=net.viktorc%3Apararealml&metric=alert_status)](https://sonarcloud.io/dashboard?id=net.viktorc%3Apararealml)
[![PyPI Version](https://badge.fury.io/py/pararealml.svg)](https://badge.fury.io/py/pararealml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pararealml.readthedocs.io/en/latest/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

PararealML is a differential equation solver library featuring a [Parareal](https://en.wikipedia.org/wiki/Parareal) framework based on a unified interface for initial value problems and various solvers including a number of machine learning accelerated ones. The library's main purpose is to provide a toolset to investigate the properties of the Parareal algorithm, especially when using machine learning accelerated coarse operators, across various problems. The library's implementation of the Parareal algorithm uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) via [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

## Main components
The library has a modular design to allow for easy extensibility. The main components that enable this are described below.

<img src="https://user-images.githubusercontent.com/12938964/156269280-58213998-58c2-45f1-aabb-7441ac4aa33e.png" alt="pararealml"/>

### Differential equation
Differential equations extend the `DifferentialEquation` base class. In PararealML, all differential equations are time-dependent. Moreover, the unknown variables of differential equations are by definition vector-valued to support systems of differential equations. Both ordinary and partial differential equations are supported.

The library provides out-of-the-box implementations for a number of differential equations including:

 * `PopulationGrowthEquation`
 * `LotkaVolterraEquation`
 * `LorenzEquation`
 * `SIREquation`
 * `VanDerPolEquation`
 * `NBodyGravitationalEquation`
 * `DiffusionEquation`
 * `ConvectionDiffusionEquation`
 * `WaveEquation`
 * `CahnHilliardEquation`
 * `BurgersEquation`
 * `ShallowWaterEquation`
 * `NavierStokesEquation`

To solve other differential equations, the `DifferentialEquation` class can be easily extended. The only method that needs to be implemented to do so is `symbolic_equation_system` which defines the system of differential equations using symbolic expressions.

### Mesh
Meshes in PararealML are represented by the `Mesh` class which defines a hyper-grid with a pair of boundaries and a uniform step size along each dimension. Meshes may be defined in a Cartesian or curvilinear (polar, cylindrical, or spherical) coordinate system.

### Boundary conditions
All boundary conditions extend the `BoundaryCondition` base class. In PararealML, boundary conditions are functions of time and space. It is possible to specify static boundary conditions that only depend on the spatial coordinates to enable the pre-computation of boundary values and thus potentially improve the performance of some of the solvers.

The most important boundary conditions provided are:

 * `DirichletBoundaryCondition`
 * `NeumannBoundaryCondition`
 * `CauchyBoundaryCondition`
 
### Constrained problem
Constrained problems encapsulate either simple ordinary differential equations or partial differential equations coupled with a mesh and boundary conditions. This offers a level of abstraction over the two kinds of differential equations. Constrained problems are represented by the `ConstrainedProblem` class whose constructor takes a differential equation and optionally a mesh and a set of boundary conditions.

### Initial conditions
In PararealML, all initial conditions extend the `InitialCondition` base class.

The library provides a number initial condition implementations including:

 * `DiscreteInitialCondition`
 * `ContinuousInitialCondition`
    * `GaussianInitialCondition`
    * `MarginalBetaProductInitialCondition`

### Initial value problem
Initial value problems are constrained problems associated with a time domain and initial conditions. They are represented by the `InitialValueProblem` class whose constructor takes a constrained problem, a tuple of two bounds defining the time interval, and an initial condition instance.

### Operator
All operators extend the `Operator` base class. Operators are standalone differential equation solvers that can be used as operators in the Parareal framework as well. They are generally constructed by providing a time step size, possibly along other arguments. They then can solve initial value problems and return the trajectories of the solutions with a temporal granularity defined by their time step sizes. The Parareal framework is an operator itself as well.

The list of provided operators is:

 * `ODEOperator`: an ODE solver based on the [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp) function of SciPy's `integrate` module
 * `FDMOperator`: a fast finite difference operator that can solve both ODEs and PDEs
 * `SupervisedMLOperator`: an supervised machine learning operator trained on data generated by an oracle operator that can solve any differential equation that it is trained on
 * `PhysicsInformedMLOperator`: a physics-informed machine learning operator that can be trained using variable initial conditions and that can solve both ODEs and PDEs
 * `PararealOperator`: a Parareal framework that can solve any differential equation its operators can

### Solution
The `solve` method of every operator returns an instance of the `Solution` class that supports a number of functionalities including interpolation, difference computation, and plot generation. The interpolation capability of this class allows the solution to be queried at any spatial point in addition to the cell centers or vertices of the mesh. Moreover, it also allows for the mixing of cell-center-oriented and vertex-oriented solvers as the fine and coarse operators of the Parareal framework.

### Plot
Visualizing the solutions of differential equations is supported by PararealML through the `Plot` base class which enables displaying and saving plots. There are a number of implementations of this class to visualize the solutions of various types of differential equations in any of the supported coordinate systems. The `generate_plots` method of the `Solution` class can conveniently generate all of the relevant plots based on the type of the problem solved. The library provides the following types of plots:

 * `TimePlot`: a simple y-against-t plot to visualise the solutions of systems of ODEs
 * `PhaseSpacePlot`: a 2D or 3D phase space plot for systems of 2 or 3 ODEs respectively
 * `NBodyPlot`: a 2D or 3D animated scatter plot for 2D or 3D n-body simulations
 * `SpaceLinePlot`: an animated line plot for 1D PDEs
 * `ContourPlot`: an animated contour plot for 2D PDEs
 * `SurfacePlot`: an animated 3D surface plot for 2D PDEs
 * `ScatterPlot`: an animated 3D scatter plot for 3D PDEs
 * `StreamPlot`: an animated 2D stream plot for the solution vector field of 2D PDEs
 * `QuiverPlot`: an animated 2D or 3D quiver plot for the solution vector field of 2D or 3D PDEs

A few examples of plots generated by PararealML can be seen below.

<p>
  <img src="https://user-images.githubusercontent.com/12938964/152646803-e044c86c-f631-4cf6-9f6b-d9efdb6ae8df.png" alt="lorenz" width="400"/>
  <img src="https://user-images.githubusercontent.com/12938964/152624090-cab353b4-0fe4-4d19-b9d0-71c9ff9103f5.png" alt="lorenz phase space" width="400"/> 
</p>
<p>
  <img src="https://user-images.githubusercontent.com/12938964/153720536-b4c22dc8-d112-4331-bcc5-130c22a36199.gif" alt="n-body" width="400"/>
  <img src="https://user-images.githubusercontent.com/12938964/152625732-f177fe9b-9184-404b-8737-79411e9ea7e3.gif" alt="wave 2d surface" width="400"/> 
</p>
<p>
  <img src="https://user-images.githubusercontent.com/12938964/152646506-1404a822-dbc9-481e-91ec-b1e6b5a49748.gif" alt="cahn hilliard 3d" width="400"/>
  <img src="https://user-images.githubusercontent.com/12938964/152649580-ced02c20-b95f-4ec2-bd81-d3e34b13f9a5.gif" alt="navier stokes stream" width="400"/> 
</p>

## Examples
The [examples](https://github.com/ViktorC/PararealML/tree/master/examples) folder contains a range of different examples of using the library for solving various differential equations both in serial and parallel. The scripts also include examples of using machine learning operators.

## Installation
To install PararealML, an implementation of the MPI standard must be pre-installed and the `mpicc` program must be on the search path as per the [installation guide](https://mpi4py.readthedocs.io/en/stable/install.html#using-pip) of mpi4py. With that set up, the library can be installed by running `pip install pararealml`.

## Development
 * To install the dependencies of the library, run `make install` (this requires an existing MPI installation and `mpicc`).
 * To perform linting, execute `make lint`.
 * The library uses type-hints throughout. For type checking, use the command `make type-check`.
 * The format any changed modules, run `make format`.
 * To run the unit tests, execute `make test`.
 * To run any of the example scripts using an arbitrary number of MPI processes, run `make run p={number of processes} example={name of example file without extension}` (e.g. `make run p=4 example=lorenz_parareal`).
