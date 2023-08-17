# Expectation maximization
Hosted here is code for fitting hierachical models to decision making datesets.

It uses an expectation maximization algorithm (originally based on Quentin Huys' work, Huys et al. PLoS CB 2012) for fitting RL models hierarchically, with a group level distribution, optional group-level covariates and per-subject parameters. See example.jl or example.ipynb for details.

It was supported in part by the National Center for Advancing Translational Sciences (NCATS), a component of the National Institute of Health (NIH), under award number UL1TR003017.

# Requirements
* Julia version 1.5+

# Install
You can install this (and the packages it depends on) from the github repository directly through the Julia package manager, e.g.

import Pkg
Pkg.add(url=\"https://github.com/ndawlab/em.git/")
