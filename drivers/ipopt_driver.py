#  Copyright 2019, Pedro Gomes.
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import os
import time
import shutil
import numpy as np
import ipyopt as opt
import subprocess as sp
from drivers.parallel_eval_driver import ParallelEvalDriver


# Wrapper to use with the Ipopt optimizer via IPyOpt.
class IpoptDriver(ParallelEvalDriver):
    def __init__(self):
        ParallelEvalDriver.__init__(self)

        # counters, flags
        self._funEval = 0
        self._jacEval = 0
        self._funReady = False
        self._jacReady = False

        # sizes
        self._nVar = 0
        self._nCon = 0

        # current value of the variables
        self._x = None

        # sparse indices of the constraint gradient, for now assumed to be dense
        self._sparseIndices = None

        # the optimization problem
        self._nlp = None
    #end

    # Update the problem parameters (triggers new evaluations).
    def update(self):
        for par in self._parameters: par.increment()

        self._x[()] = 1e20
        self._funReady = False
        self._jacReady = False
        self._resetAllValueEvaluations()
        self._resetAllGradientEvaluations()

        if self._hisObj is not None:
            self._hisObj.write("Parameter update.\n")
    #end

    # Prepares and returns the optimization problem for Ipopt.
    # For convenience also does other preprocessing, all functions must be set before calling this method.
    # Do not destroy the driver after obtaining the problem.
    def getNLP(self):
        self.preprocessVariables()

        self._ofval = np.zeros((len(self._objectives),))
        self._eqval = np.zeros((len(self._constraintsEQ),))
        self._ltval = np.zeros((len(self._constraintsLT),))
        self._gtval = np.zeros((len(self._constraintsGT),))
        self._inval = np.zeros((len(self._constraintsIN),))

        # write the header for the history file
        if self._hisObj is not None:
            header = "ITER"+self._hisDelim
            for obj in self._objectives:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsEQ:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsLT:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsGT:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsIN:
                header += obj.function.getName()+self._hisDelim
            header = header.strip(self._hisDelim)+"\n"
            self._hisObj.write(header)
        #end

        # initialize current values such that evaluations are triggered on first call
        self._nVar = self.getNumVariables()
        self._x = np.ones([self._nVar,])*1e20

        # prepare constraint information, the bounds are based on the shifting and scaling
        self._nCon = len(self._constraintsEQ) + len(self._constraintsLT) +\
                     len(self._constraintsGT) + len(self._constraintsIN)

        conLowerBound = np.zeros([self._nCon,])
        conUpperBound = np.zeros([self._nCon,])

        i = len(self._constraintsEQ)
        conLowerBound[i:(i+len(self._constraintsLT))] = -1e20

        i += len(self._constraintsLT)
        conUpperBound[i:(i+len(self._constraintsGT))] = 1e20

        i += len(self._constraintsGT)
        conUpperBound[i:(i+len(self._constraintsIN))] = 1.0

        # assume row major storage for gradient sparsity
        rg = range(self._nVar * self._nCon)
        self._sparseIndices = (np.array([i // self._nVar for i in rg], dtype=int),
                               np.array([i % self._nVar for i in rg], dtype=int))

        # create the optimization problem
        self._nlp = opt.Problem(self._nVar, self.getLowerBound(), self.getUpperBound(),
                                self._nCon, conLowerBound, conUpperBound, self._sparseIndices, 0,
                                self._eval_f, self._eval_grad_f, self._eval_g, self._eval_jac_g)
        return self._nlp
    #end

    # Writes a line to the history file.
    def _writeHisLine(self):
        if self._hisObj is None: return
        hisLine = str(self._funEval)+self._hisDelim
        for val in self._ofval:
            hisLine += str(val)+self._hisDelim
        for val in self._eqval:
            hisLine += str(val)+self._hisDelim
        for val in self._ltval:
            hisLine += str(val)+self._hisDelim
        for val in self._gtval:
            hisLine += str(val)+self._hisDelim
        for val in self._inval:
            hisLine += str(val)+self._hisDelim
        hisLine = hisLine.strip(self._hisDelim)+"\n"
        self._hisObj.write(hisLine)
    #end

    # Detect a change in the design vector, reset directories and evaluation state.
    def _handleVariableChange(self,x):
        assert x.size == self._nVar, "Wrong size of design vector."

        newValues = (abs(self._x-x) > np.finfo(float).eps).any()

        if not newValues: return False

        # otherwise...

        # update the values of the variables
        self._setCurrent(x)
        self._x[()] = x

        # manage working directories
        os.chdir(self._userDir)
        if os.path.isdir(self._workDir):
            if self._keepDesigns:
                dirName = self._dirPrefix+str(self._funEval).rjust(3,"0")
                if os.path.isdir(dirName): shutil.rmtree(dirName)
                os.rename(self._workDir,dirName)
            else:
                shutil.rmtree(self._workDir)
            #end
        #end
        os.mkdir(self._workDir)

        # trigger evaluations
        self._funReady = False
        self._jacReady = False
        self._resetAllValueEvaluations()
        self._resetAllGradientEvaluations()

        return True
    #end

    # Method passed to Ipopt to get the objective value,
    # evaluates all functions if necessary.
    def _eval_f(self,x):
        if self._handleVariableChange(x):
            self._evaluateFunctions()
        #end

        return self._ofval.sum()
    #end

    # Method passed to Ipopt to get the objective gradient, evaluates gradients and
    # functions if necessary, otherwise it simply combines and scales the results.
    def _eval_grad_f(self, x, out):
        assert out.size >= self._nVar, "Wrong size of gradient vector (\"out\")."

        if self._handleVariableChange(x):
            self._evaluateGradients()
        #end

        self._jacTime -= time.time()
        os.chdir(self._workDir)

        out[()] = 0.0
        for obj in self._objectives:
            out += obj.function.getGradient(self._variableStartMask) * obj.scale
        out /= self._varScales

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return out
    #end

    # Method passed to Ipopt to expose the constraint vector, see also "_eval_f"
    def _eval_g(self, x, out):
        assert out.size >= self._nCon, "Wrong size of constraint vector (\"out\")."

        if self._handleVariableChange(x):
            self._evaluateFunctions()
        #end

        i = 0
        out[i:(i+len(self._constraintsEQ))] = self._eqval

        i += len(self._constraintsEQ)
        out[i:(i+len(self._constraintsLT))] = self._ltval

        i += len(self._constraintsLT)
        out[i:(i+len(self._constraintsGT))] = self._gtval

        i += len(self._constraintsGT)
        out[i:(i+len(self._constraintsIN))] = self._inval

        return out
    #end

    # Method passed to Ipopt to expose the constraint Jacobian, see also "_eval_grad_f".
    def _eval_jac_g(self, x, out):
        assert out.size >= self._nCon*self._nVar, "Wrong size of constraint Jacobian vector (\"out\")."

        if self._handleVariableChange(x):
            self._evaluateGradients()
        #end

        self._jacTime -= time.time()
        os.chdir(self._workDir)

        i = 0
        mask = self._variableStartMask

        for conType in [self._constraintsEQ, self._constraintsLT,\
                        self._constraintsGT, self._constraintsIN]:
            for con in conType:
                out[i:(i+self._nVar)] = con.function.getGradient(mask) * con.scale / self._varScales
                i += self._nVar
            #end
        #end

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return out
    #end

    # Evaluate all functions (objectives and constraints), imediately
    # retrieves and stores the results after shifting and scaling.
    def _evaluateFunctions(self):
        # lazy evaluation
        if self._funReady: return

        if self._userPreProcessFun:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessFun,shell=True)
        #end

        os.chdir(self._workDir)

        # evaluate everything, either in parallel or sequentially
        if self._parallelEval: self._evalFunInParallel()

        self._funEval += 1
        self._funTime -= time.time()

        for i in range(self._ofval.size):
            self._ofval[i] = self._objectives[i].function.getValue()

        for i in range(self._eqval.size):
            self._eqval[i] = self._constraintsEQ[i].function.getValue()

        for i in range(self._ltval.size):
            self._ltval[i] = self._constraintsLT[i].function.getValue()

        for i in range(self._gtval.size):
            self._gtval[i] = self._constraintsGT[i].function.getValue()

        for i in range(self._inval.size):
            self._inval[i] = self._constraintsIN[i].function.getValue()

        self._funTime += time.time()

        # monitor convergence (raw function values)
        self._writeHisLine()

        # shift constraints and scale as required
        for i in range(self._ofval.size):
            self._ofval[i] *= self._objectives[i].scale

        for i in range(self._eqval.size):
            self._eqval[i] -= self._constraintsEQ[i].bound1
            self._eqval[i] *= self._constraintsEQ[i].scale

        for i in range(self._ltval.size):
            self._ltval[i] -= self._constraintsLT[i].bound1
            self._ltval[i] *= self._constraintsLT[i].scale

        for i in range(self._gtval.size):
            self._gtval[i] -= self._constraintsGT[i].bound1
            self._gtval[i] *= self._constraintsGT[i].scale

        for i in range(self._inval.size):
            self._inval[i] -= self._constraintsIN[i].bound1
            self._inval[i] *= self._constraintsIN[i].scale

        os.chdir(self._userDir)

        self._funReady = True
    #end

    # Evaluates all gradients in parallel execution mode, otherwise
    # it only runs the user preprocessing and the execution takes place
    # when the results are read in "_eval_grad_f" or in "_eval_jac_g".
    def _evaluateGradients(self):
        # lazy evaluation
        if self._jacReady: return

        # we assume that evaluating the gradients requires the functions
        self._evaluateFunctions()

        if self._userPreProcessGrad:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessGrad,shell=True)
        #end

        os.chdir(self._workDir)

        # evaluate everything, either in parallel or sequentially,
        # in the latter case the evaluations occur when retrieving the values
        if self._parallelEval: self._evalJacInParallel()

        os.chdir(self._userDir)

        self._jacEval += 1
        self._jacReady = True
    #end
