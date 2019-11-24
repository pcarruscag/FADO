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
evalFun1 = ExternalRun("RUN1","python ../../direct.py config_tmpl.txt")
evalFun1.addConfig("config_tmpl.txt")
evalFun1.addData("data1.txt")
evalFun1.addParameter(parData1)

evalJac1 = ExternalRun("JAC1","python ../../adjoint.py config_tmpl.txt")
evalJac1.addConfig("config_tmpl.txt")
evalJac1.addData("data1.txt")
evalJac1.addData("RUN1/results.txt") # simulate we need data from the direct run
evalJac1.addParameter(parData1)
evalJac1.addParameter(parFunc1)

evalFun2 = ExternalRun("RUN2","python ../../direct.py config_tmpl.txt")
evalFun2.addConfig("config_tmpl.txt")
evalFun2.addData("data2.txt")
evalFun2.addParameter(parData2)

evalJac2 = ExternalRun("JAC2","python ../../adjoint.py config_tmpl.txt")
evalJac2.addConfig("config_tmpl.txt")
evalJac2.addData("data2.txt")
evalJac2.addData("RUN2/results.txt") # simulate we need data from the direct run
evalJac2.addParameter(parData2)
evalJac2.addParameter(parFunc1)

evalJac3 = ExternalRun("JAC3","python ../../adjoint.py config_tmpl.txt")
evalJac3.addConfig("config_tmpl.txt")
evalJac3.addData("data2.txt")
evalJac3.addData("RUN2/results.txt") # simulate we need data from the direct run
evalJac3.addParameter(parData2)
evalJac3.addParameter(parFunc2)

# Functions
# now variables, parameters, and evaluations are combined
fun1 = Function("Rosenbrock1","RUN1/results.txt",TableReader(0,0))
fun1.addInputVariable(var1,"JAC1/gradient.txt",TableReader(0,0))
fun1.addInputVariable(var2,"JAC1/gradient.txt",TableReader(1,0))
fun1.addValueEvalStep(evalFun1)
fun1.addGradientEvalStep(evalJac1)

fun2 = Function("Rosenbrock2","RUN2/results.txt",TableReader(0,0))
fun2.addInputVariable(var1,"JAC2/gradient.txt",TableReader(0,0))
fun2.addInputVariable(var2,"JAC2/gradient.txt",TableReader(1,0))
fun2.addValueEvalStep(evalFun2)
fun2.addGradientEvalStep(evalJac2)

fun3 = Function("Constraint2","RUN2/results.txt",TableReader(1,0))
fun3.addInputVariable(var1,"JAC3/gradient.txt",TableReader(0,0))
fun3.addInputVariable(var2,"JAC3/gradient.txt",TableReader(1,0))
fun3.addValueEvalStep(evalFun2)
fun3.addGradientEvalStep(evalJac3)

# Driver
# the optimization is defined by the objectives and constraints
driver = ExteriorPenaltyDriver(0.005)
driver.addObjective("min",fun1,0.5)
driver.addObjective("min",fun2,0.5)
driver.addUpperBound(fun3,2.0)

driver.preprocessVariables()

# now by calling driver.fun or driver.grad the driver should do all the work
