# PararealML [![Build Status](https://travis-ci.org/ViktorC/PararealML.svg?branch=master)](https://travis-ci.org/ViktorC/PararealML) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=net.viktorc%3Apararealml&metric=alert_status)](https://sonarcloud.io/dashboard?id=net.viktorc%3Apararealml) [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pararealml.readthedocs.io/en/latest/index.html) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

PararealML is a differential equation solver library featuring a [Parareal](https://en.wikipedia.org/wiki/Parareal) framework based on a unified interface for various solvers including a range of machine learning accelerated ones. The library's main purpose is to provide a toolset to investigate the properties of the Parareal algorithm, especially when using machine learning accelerated coarse operators, across various problems. The library's implementation of the Parareal algorithm uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) via [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

## Main components

The library has a modular design to allow for easy extensibility. The main components that enable this are described below.

### Differential equation

All differential equations extend the `DifferentialEquation` base class. In PararealML, all differential equations are time-dependent. Moreover, the unknown variables of differential equations are by definition vector-valued to support systems of differential equations. Both ordinary and partial differential equations are supported.

The library provides out-of-the-box implementations for a number of differential equations including:

 * `PopulationGrowthEquation`
 * `LotkaVolterraEquation`
 * `LorenzEquation`
 * `NBodyGravitationalEquation` - 2 or 3D
 * `DiffusionEquation` - n-D
 * `WaveEquation` - n-D
 * `CahnHilliardEquation` - n-D
 * `NavierStokesEquation` - 2 or 3D

To solve other differential equations, the `DifferentialEquation` class can be easily extended. Depending on the operators intended to be used for solving the differential equation, not all of the abstract methods may need to be implemented.

### Mesh

All meshes extend the `Mesh` base class. Meshes define the spatial domains of initial boundary value problems and also the discretisations of these domains.

The library currently only supports uniform grids through the `UniformGrid` class (of any dimensions for the `FDMOperator` and for up to 3 dimensions for the `FVMOperator` and the `PINNOperator`) but the `Mesh` class can be easily extended to support more complex meshes.

### Boundary conditions

All boundary conditions extend the `BoundaryCondition` base class. In PararealML, boundary conditions are currently functions of the spatial coordinates only (to enable the pre-computation of boundary values for better performance).

The list of supported boundary conditions is:

 * `DirichletBoundaryCondition`
 * `NeumannBoundaryCondition`
 * `CauchyBoundaryCondition`
 
To implement other types of boundary conditions, the `BoundaryCondition` class may be extended.

### Constrained problem

Constrained problems encapsulate either simple ordinary differential equations or partial differential equations coupled with a mesh and boundary conditions. This offers a level of abstraction over the two kinds of differential equations. Constrained problems are represented by the `ConstrainedProblem` class whose constructor takes a differential equation and optionally a mesh and a set of boundary conditions.

### Initial conditions

In PararealML, all initial conditions extend the `InitialCondition` base class.

The library provides a number initial condition implementations including:

 * `DiscreteInitialCondition` - discrete initial conditions defined by a NumPy array and whether they are vertex or cell-center oriented
 * `ContinuousInitialCondition` - continuous initial conditions defined by a function
    * `GaussianInitialCondition` - a hump defined by a multivariate Gaussian distribution
 
To implement other types of boundary conditions, the `InitialCondition` class may be extended.

### Initial value problem

Initial value problems are constrained problems associated with a time domain and initial conditions. They are represented by the `InitialValueProblem` class whose constructor takes a constrained problem, a tuple of two `float`s defining the time interval, and an initial condition instance.

### Operator

All operators extend the `Operator` base class. Operators are standalone differential equation solvers that can be used as operators in the Parareal framework as well. They are generally constructed by giving them a time step size. They then can solve initial value problems and return the trajectories of the solutions with a granularity defined by their time step sizes. The Parareal framework is an operator itself as well.

The list of supported operators is:

 * `ODEOperator` - based on the [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp) function of SciPy's `integrate` module, can solve all ordinary differential equations
 * `FVMOperator` - finite volume method operator based on [FiPy](https://www.ctcms.nist.gov/fipy/), can solve all partial differential equations except for Navier-Stokes
 * `FDMOperator` - a self-implemented fast finite difference method operator, can solve any differential equation
 * `MLOperator`
    * `StatelessMLOperator`
        * `PINNOperator` - a physics-informed neural network operator based on `DeepXDE`, can solve all differential equations except for n-body and Navier-Stokes
        * `StatelessRegressionOperator` - a regression operator trained on data generated by another operator, can solve any differential equation the trainer operator can
    * `StatefulMLOperator`
        * `StatefulRegressionOperator` - a regression operator trained on data generated by another operator taking the previous value of the solution into account, can solve any differential equation the trainer operator can
 * `PararealOperator` - a Parareal framework that can solve any differential equation its operators can

## Solution

The `solve` method of every operator returns an instance of the `Solution` class that supports a number of functionalities including interpolation and plotting.

### Visualisation

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

On the other hand, the solutions of 2D partial differential equations can be visualised in two different ways. The first one of these is a 3D surface plot.

<img src="https://user-images.githubusercontent.com/12938964/91648397-fc5cff80-ea5e-11ea-887d-187523ae701b.gif" alt="wave_2d" width="400"/>

Finally, the second way of visualising the solution of 2D partial differential equations is the generation of a 2D contour plot.

<img src="https://user-images.githubusercontent.com/12938964/91649018-dc313e80-ea66-11ea-9995-dd98af8dbdac.gif" alt="navier_stokes_2d" width="400"/>

## Examples

The [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder contains a range of different examples of using the library for solving various differential equations both in serial and parallel. The scripts also include examples of using machine learning operators.

## Setup
 1. make sure you have a working Anaconda installation (see the [guide](https://docs.anaconda.com/anaconda/install/))
 1. `conda create -n {environment_name} python={python_version}` - replace `{environment_name}` with the name of your environment and replace `{python_version}` with any version number greater than or equal to `3.7`
 1. `conda activate {environment_name}`
 1. `conda install -c conda-forge mpi` - not necessary if there is already an implementation of the MPI standard installed (e.g. [MPICH](https://www.mpich.org/))
 1. `conda install -c conda-forge imagemagick` - only used for saving animated plots; not necessary if `imagemagick` is already installed
 1. `conda install -c conda-forge fipy==3.3`
 1. `make install`

## Testing

To perform linting, execute `make lint`.

The library uses type-hints throughout. For type checking, use the command `make type-check`.

To run the unit tests, execute `make test`.

## Running

To run any of the examples from the [examples](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/tree/master/code/python/examples) folder using 4 MPI processes, run `make run example={name of example file without extension}` (e.g. `make run example=diffusion_1d_serial`).
