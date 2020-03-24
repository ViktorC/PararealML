# Parallel-in-time methods with ML

## Project information

**Student name:** Viktor Csomor

**Student number:** s1984842

**EPCC supervisor name(s):** Rupert Nash, Anna Roubickova

## Description

Differential equations describe numerous phenomena that play important roles in science, engineering, and finance. Several different numerical methods can be used to solve differential equations based on their properties. Due to the prevalence of these equations, a lot of effort has been put into parallelising their solution. One wall clock time efficient way of doing this for time dependent, fixed sized problems is parallelising across the time steps. The best known parallel-in-time method is the parareal algorithm which relies on the iterative application of a course operator _G_ to approximate the solution serially followed by a fine operator _F_ to refine the estimate. This project is concerned with the training of a machine learning (ML) model and using it as the operator _G_ in a parareal framework.

The hypotheses of the project are that:
* an ML model can be trained and used as the course operator _G_
* for a given level of accuracy, the trained ML model can outperform a traditional operator in terms of wall clock time for some problems
* for a given number of CPU (and GPU) cores, the ML accelerated parareal framework outperforms a traditional parallel solver in terms of wall clock time for some problems

The original project proposal: [link](https://www.wiki.ed.ac.uk/pages/viewpage.action?spaceKey=hpcdis&title=Parallel-in-time+methods+with+ML)

## Content

The repository contains the following:

* Preliminary work:
    * Code:
        * C:
            * The serial solution of a 1D diffusion equation by discretising along the spatial dimension using the finite difference method and applying the Euler method to the resulting system of ODEs
            * The parallel solution of the rabbit population differential equation in a parareal framework built on top of POSIX threads using the Euler method as _G_ and RK4 as _F_
        * Python:
            * A port of the C code to Python using `mpi4py` allowing for different time step sizes across the two operators _G_ and _F_
            * A simple machine learning accelerated operator as _G_
* Project presentation
* Report
* Minutes from supervisor meetings and other notes (in the Wiki)
