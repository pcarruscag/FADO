# Plain topology optimization example

import sys
sys.path.append("../")
sys.path.append("../../")

from FADO import *
import scipy.optimize
import subprocess
subprocess.call("unzip -o data.zip",shell=True)

# Design variables
rho = InputVariable(0.5,TableWriter("  ",(1,-1)),1600,1.0,0.0,1.0)

# Parameters
pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
fType_objective = Parameter(["REFERENCE_NODE"],LabelReplacer("__FUNCTION__"))
fType_constraint = Parameter(["VOLUME_FRACTION"],LabelReplacer("__FUNCTION__"))
beta = Parameter([0.01, 1, 4, 16, 64, 200],LabelReplacer("__BETA__"))

# Evaluations
directRun = ExternalRun("DIRECT","SU2_CFD settings_tmpl.cfg")
directRun.addConfig("settings_tmpl.cfg")
directRun.addConfig("element_properties.dat")
directRun.addData("mesh.su2")
directRun.addParameter(pType_direct)
directRun.addParameter(fType_objective)
directRun.addParameter(beta)

adjointRun1 = ExternalRun("ADJOINT1","SU2_CFD_AD settings_tmpl.cfg")
adjointRun1.addConfig("settings_tmpl.cfg")
adjointRun1.addConfig("element_properties.dat")
adjointRun1.addData("mesh.su2")
adjointRun1.addData("DIRECT/direct.dat")
adjointRun1.addParameter(pType_adjoint)
adjointRun1.addParameter(fType_objective)
adjointRun1.addParameter(beta)

adjointRun2 = ExternalRun("ADJOINT2","SU2_CFD_AD settings_tmpl.cfg")
adjointRun2.addConfig("settings_tmpl.cfg")
adjointRun2.addConfig("element_properties.dat")
adjointRun2.addData("mesh.su2")
adjointRun2.addData("DIRECT/direct.dat")
adjointRun2.addParameter(pType_adjoint)
adjointRun2.addParameter(fType_constraint)
adjointRun2.addParameter(beta)

# Functions
fun1 = Function("reference_node","DIRECT/of_refnode.dat",TableReader(0,0))
fun1.addInputVariable(rho,"ADJOINT1/grad.dat",TableReader(None,0))
fun1.addValueEvalStep(directRun)
fun1.addGradientEvalStep(adjointRun1)

fun2 = Function("solid_fraction","ADJOINT2/of_volfrac.dat",TableReader(0,0))
fun2.addInputVariable(rho,"ADJOINT2/grad.dat",TableReader(None,0))
fun2.addValueEvalStep(directRun)
fun2.addValueEvalStep(adjointRun2)

# Driver
update_iters = 25
driver = ExteriorPenaltyDriver(0.005,update_iters)
driver.addObjective("min",fun1,80.0)
driver.addUpperBound(fun2,0.5,2.0)

driver.preprocessVariables()
driver.setEvaluationMode(True,1.0)
driver.setStorageMode(False)

# Optimization
x  = driver.getInitial()
lb = driver.getLowerBound()
ub = driver.getUpperBound()
bounds = np.array((lb,ub),float).transpose()
max_iters = 200
options={'disp': True, 'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-12, 'maxiter': update_iters}

while not beta.isAtTop():
  optimum = scipy.optimize.minimize(driver.fun, x, method="L-BFGS-B",\
            jac=driver.grad, bounds=bounds, options=options)
  x = optimum.x
  max_iters -= optimum.nit
#end

options['maxiter'] = max_iters
optimum = scipy.optimize.minimize(driver.fun, x, method="L-BFGS-B",\
          jac=driver.grad, bounds=bounds, options=options)

