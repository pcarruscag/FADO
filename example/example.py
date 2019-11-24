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

