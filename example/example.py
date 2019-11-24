# This example will serve to define the "interface" of the framework.
# It should mimic the way an external code works so that it can drive the development
# and highlight problems with the design as early as possible.
#
# For generality:
# 3 functions that result from 2 evaluation of the primal function
#
# Files:
#  - direct.py is the primal "solver", it takes a data file and a config.
#    Two data files will form the two evaluations, it computes 2 functions.
#  - adjoint.py computes the gradient of the requested function. 
#
# The variables (x,y) go in the config the constants go in the data.
#

import sys
sys.path.append("../")
sys.path.append("../../")

from FADO import *

# Design variables of the problem
# this defines initial value and how they are written to an arbitrary file
var1 = InputVariable(0.0,LabelReplacer("__X__"))
var2 = InputVariable(0.0,LabelReplacer("__Y__"))

# Parameters
# these parameters taylor the template config to each function
parData1 = Parameter(["data1.txt"],LabelReplacer("__DATA_FILE__"))
parData2 = Parameter(["data2.txt"],LabelReplacer("__DATA_FILE__"))
parFunc1 = Parameter(["rosenbrock"],LabelReplacer("__FUNCTION__"))
parFunc2 = Parameter(["constraint"],LabelReplacer("__FUNCTION__"))

# Evaluations
# "runs" that are needed to compute functions and their gradients
evalRun1 = 


# Ouput variables
# these will define objectives and constraints
