# Parallel-in-time methods with ML

## Project information

**Student name:** Viktor Csomor

**Student number:** s1984842

**EPCC supervisor name(s):** Rupert Nash, Anna Roubickova

## Description

Differential equations describe numerous phenomena that play important roles in science, engineering, and finance. Several different numerical methods can be used to solve differential equations based on their properties. Due to the prevalence of these equations, a lot of effort has been put into parallelising their solution. One wall clock time efficient way of doing this for time dependent, fixed sized problems is parallelising across time. The best known parallel-in-time method is the Parareal algorithm which relies on the iterative application of a course operator _G_ to approximate the solution serially followed by a fine operator _F_ to refine these estimates. This project is concerned with the training of a machine learning  model and using it as the operator _G_ in a parareal framework.

The original project proposal: [link](https://www.wiki.ed.ac.uk/pages/viewpage.action?spaceKey=hpcdis&title=Parallel-in-time+methods+with+ML)

## Content

The repository contains the following:

* PararealML: a Python library providing a Parareal framework based on a unified interface for various differential equation solvers including a finite volume method solver, a PINN solver, and its own finite difference method solver (for more detail, see the library's [README](https://git.ecdf.ed.ac.uk/msc-19-20/s1984842/blob/master/code/python/README.md))
* Project presentation
* Report
* Results of the dissertation project experiments
* Minutes from supervisor meetings and other notes (in the Wiki)
