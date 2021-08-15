# PararealML [![Build Status](https://travis-ci.com/ViktorC/PararealML.svg?branch=master)](https://travis-ci.com/ViktorC/PararealML) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=net.viktorc%3Apararealml&metric=alert_status)](https://sonarcloud.io/dashboard?id=net.viktorc%3Apararealml) [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pararealml.readthedocs.io/en/latest/index.html) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

PararealML is a differential equation solver library featuring a [Parareal](https://en.wikipedia.org/wiki/Parareal) framework based on a unified interface for initial value problems and various solvers including a number of machine learning accelerated ones. The library's main purpose is to provide a toolset to investigate the properties of the Parareal algorithm, especially when using machine learning accelerated coarse operators, across various problems. The library's implementation of the Parareal algorithm uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) via [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

## Main components
The library has a modular design to allow for easy extensibility. The main components that enable this are described below.

### Differential equation
Differential equations extend the `DifferentialEquation` base class. In PararealML, all differential equations are time-dependent. Moreover, the unknown variables of differential equations are by definition vector-valued to support systems of differential equations. Both ordinary and partial differential equations are supported.

The library provides out-of-the-box implementations for a number of differential equations including:

 * `PopulationGrowthEquation`
 * `LotkaVolterraEquation`
 * `LorenzEquation`
 * `NBodyGravitationalEquation`
 * `DiffusionEquation`
 * `ConvectionDiffusionEquation`
 * `WaveEquation`
 * `CahnHilliardEquation`
 * `BurgerEquation`
 * `ShallowWaterEquation`
 * `NavierStokesStreamFunctionVorticityEquation`

To solve other differential equations, the `DifferentialEquation` class can be easily extended. The only method that needs to be implemented to do so is `symbolic_equation_system` which defines the system of differential equations using symbolic expressions.

### Mesh
Meshes in PararealML are represented by the `Mesh` class which defines a hyper-grid with a pair of boundaries and a uniform step size along each dimension. Meshes may be defined in a Cartesian or curvilinear (polar, cylindrical, or spherical) coordinate system. However, there is no guarantee that all operators will support all coordinate system types.

### Boundary conditions
All boundary conditions extend the `BoundaryCondition` base class. In PararealML, boundary conditions are functions of time and space. It is possible to specify static boundary conditions that only depend on the spatial coordinates to enable the pre-computation of boundary values and thus potentially improve the performance of some of the solvers).

The list of provided boundary conditions is:

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
 
To implement other types of boundary conditions, the `InitialCondition` class may be easily extended.

### Initial value problem
Initial value problems are constrained problems associated with a time domain and initial conditions. They are represented by the `InitialValueProblem` class whose constructor takes a constrained problem, a tuple of two `float`s defining the time interval, and an initial condition instance.

### Operator
All operators extend the `Operator` base class. Operators are standalone differential equation solvers that can be used as operators in the Parareal framework as well. They are generally constructed by providing a time step size, possibly along other arguments. They then can solve initial value problems and return the trajectories of the solutions with a temporal granularity defined by their time step sizes. The Parareal framework is an operator itself as well.

The list of supported operators is:

 * `ODEOperator` - an ODE solver based on the [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp) function of SciPy's `integrate` module
 * `FDMOperator` - a fast Finite Difference Method based operator that can solve both ODEs and PDEs
 * `PIDONOperator` - a Physics-Informed DeepONet operator (based on [https://arxiv.org/abs/2103.10974](https://arxiv.org/abs/2103.10974)) which can be trained using variable initial conditions and that can solve both ODEs and PDEs
 * `AutoRegressionOperator` - an auto-regression operator trained on data generated by an oracle operator that can solve any differential equation similar to the one it was trained on
 * `PararealOperator` - a Parareal framework that can solve any differential equation its operators can

### Solution
The `solve` method of every operator returns an instance of the `Solution` class that supports a number of functionalities including interpolation and plotting. The interpolation capability of this class allows the solution to be queried at any spatial point in addition to the cell centers or vertices of the mesh. Moreover, it allows for the mixing of cell-center-oriented and vertex-oriented solvers as the fine and coarse operators of the Parareal framework.

## Visualisation
The visualisation of solutions is facilitated by the `plot` method of the `Solution` class that generates plots depending on the type of differential equation solved.

#### Ordinary differential equations
The solutions of ordinary differential equations are plotted as y-against-t plots. In the case of ordinary differential equations, the phase space plot is generated as well.

<p float="left">
  <img src="https://user-images.githubusercontent.com/12938964/91643889-07983700-ea2f-11ea-8553-573a16d96d5f.jpg" alt="lorenz" width="400"/>
  <img src="https://user-images.githubusercontent.com/12938964/91643891-0c5ceb00-ea2f-11ea-854d-0813de46d520.jpg" alt="lorenz_phase_space" width="400"/> 
</p>

#### N-body simulations
The solutions of n-body simulations are plotted using a special 2D or 3D scatter plot (depending on the dimensionality of the simulation).

<img src="https://user-images.githubusercontent.com/12938964/91643717-8f7d4180-ea2d-11ea-834b-7f64e8347557.gif" alt="n_body" width="400"/>

#### Partial differential equations in one spatial dimension
The solutions of 1D partial differential equations are visualised as animated line plots.

<img src="https://user-images.githubusercontent.com/12938964/91645244-6ebbe880-ea3b-11ea-925d-54b7a3e0bfee.gif" alt="diffusion_1d" width="400"/>

#### Partial differential equations in two spatial dimensions
On the other hand, the solutions of 2D partial differential equations can be visualised in multiple different ways. The first one of these is a 3D surface plot.

<img src="https://user-images.githubusercontent.com/12938964/91648397-fc5cff80-ea5e-11ea-887d-187523ae701b.gif" alt="wave_2d" width="400"/>

The second way is 2D contour plots.

<img src="https://user-images.githubusercontent.com/12938964/96491705-8133f080-123a-11eb-98fa-7f4b43730ac1.gif" alt="navier_stokes_2d" width="400"/>

Finally, 2D vector fields can be visualised as 2D quiver plots.

<img src="https://user-images.githubusercontent.com/12938964/96521862-231f0180-126a-11eb-948f-718117b63895.gif" alt="shallow_water_2d" width="400"/>

#### Partial differential equations in three spatial dimensions
The solutions of partial differential equation in three spatial dimensions can be visualised in two different ways. The first one of these is 2D contour plots of slices of the solution along an arbitrary axis at arbitrary points.

<p float="left">
  <img src="https://user-images.githubusercontent.com/12938964/97097322-369ce480-163d-11eb-9ccc-7505bca2b2b2.gif" alt="cahn_hilliard_slice_0" width="266"/>
  <img src="https://user-images.githubusercontent.com/12938964/97097323-37357b00-163d-11eb-9258-bd699ef7ea29.gif" alt="cahn_hilliard_slice_1" width="266"/>
  <img src="https://user-images.githubusercontent.com/12938964/97097325-37ce1180-163d-11eb-8acb-b6d996ea178a.gif" alt="cahn_hilliard_slice_2" width="266"/>
</p>

The second way, only applicable to 3D vector fields, is the use of 3D quiver plots.

<img src="https://user-images.githubusercontent.com/12938964/97097783-1708ba80-1643-11eb-8349-f065de5ddc4d.gif" alt="burger_3d" width="400"/>

## Examples
The [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder contains a range of different examples of using the library for solving various differential equations both in serial and parallel. The scripts also include examples of using machine learning operators.

## Setup
To use the Parareal operator, an implementation of the MPI standard must be installed (e.g. [MPICH](https://www.mpich.org/)). To save animated plots, [ImageMagick](https://imagemagick.org/index.php) must be installed. These programs can be easily installed using [Anaconda](https://www.anaconda.com/). If they are already installed and available on the system, you can skip to step 6 of the setup guide.
 1. make sure you have a working Anaconda installation (see the [guide](https://docs.anaconda.com/anaconda/install/))
 1. `conda create -n {environment_name} python={python_version}` - replace `{environment_name}` with the name of your environment and replace `{python_version}` with any version number greater than or equal to `3.8`
 1. `conda activate {environment_name}`
 1. `conda install -c conda-forge mpi4py`
 1. `conda install -c conda-forge imagemagick`
 1. `make install`
 
 On Windows, steps 1 to 5 do not apply. Instead, both Microsoft MPI and ImageMagick must be installed and added to the PATH manually.

## Testing
To perform linting, execute `make lint`.

The library uses type-hints throughout. For type checking, use the command `make type-check`.

To run the unit tests, execute `make test`.

## Running
To run any of the examples from the [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder using 4 MPI processes, run `make run example={name of example file without extension}` (e.g. `make run example=diffusion_1d_fdm`).
