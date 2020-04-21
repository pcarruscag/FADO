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
import copy
import shutil
import numpy as np
import subprocess as sp
from drivers.parallel_eval_driver import ParallelEvalDriver


# Exterior Penalty method wrapper
class ExteriorPenaltyDriver(ParallelEvalDriver):
    def __init__(self, tol, freq=40, rini=8, rmax=1024, factorUp=4, factorDown=0.5):
        ParallelEvalDriver.__init__(self, True)
        
        # parameters of the method
        self._tol = tol
        self._freq = freq
        self._rini = rini
        self._rmax = rmax
        self._cup = factorUp
        self._cdown = factorDown

        # constraint penalties
        self._eqpen = None
        self._ltpen = None
        self._gtpen = None
        self._inpen = None

        # gradient vector
        self._grad = None
        self._old_grad = None

        # timers, counters, flags
        self._funEval = 0
        self._jacEval = 0
        self._isInit = False
        self._isFeasible = False
        self._logRowFormat = ""
    #end

    # method for lazy initialization
    def _initialize(self):
        if self._isInit: return

        self._ofval = np.zeros((len(self._objectives),))
        self._eqval = np.zeros((len(self._constraintsEQ),))
        self._ltval = np.zeros((len(self._constraintsLT),))
        self._gtval = np.zeros((len(self._constraintsGT),))
        self._inval = np.zeros((len(self._constraintsIN),))

        self._eqpen = np.ones((len(self._constraintsEQ),))*self._rini
        self._ltpen = np.ones((len(self._constraintsLT),))*self._rini
        self._gtpen = np.ones((len(self._constraintsGT),))*self._rini
        self._inpen = np.ones((len(self._constraintsIN),))*self._rini

        self._grad = np.zeros((self.getNumVariables(),))
        self._old_grad = copy.deepcopy(self._grad)

        # write the header for the log file and set the format
        if self._logObj is not None:
            w = self._logColWidth
            headerData = ["FUN EVAL","FUN TIME","GRAD EVAL","GRAD TIME","FEASIBLE"]
            self._logRowFormat = "{:>W}"+"{:>W.3e}{:>W}"*2
            for obj in self._objectives:
                headerData.append(obj.function.getName(w-1))
                self._logRowFormat += "{:>W.Pg}"
            for obj in self._constraintsEQ:
                headerData.append(obj.function.getName(w-1))
                headerData.append("PEN COEFF")
                self._logRowFormat += "{:>W.Pg}"*2
            for obj in self._constraintsLT:
                headerData.append(obj.function.getName(w-1))
                headerData.append("PEN COEFF")
                self._logRowFormat += "{:>W.Pg}"*2
            for obj in self._constraintsGT:
                headerData.append(obj.function.getName(w-1))
                headerData.append("PEN COEFF")
                self._logRowFormat += "{:>W.Pg}"*2
            for obj in self._constraintsIN:
                headerData.append(obj.function.getName(w-1))
                headerData.append("PEN COEFF")
                self._logRowFormat += "{:>W.Pg}"*2
            # right-align, set width in format and a precision that fits it
            self._logRowFormat = self._logRowFormat.replace("W",str(w))+"\n"
            self._logRowFormat = self._logRowFormat.replace("P",str(min(8,w-7)))
            header = ""
            for data in headerData:
                header += data.rjust(w)
            self._logObj.write(header+"\n")
        #end

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

        self._isInit = True
    #end

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

    def _writeLogLine(self):
        if self._logObj is None: return
        data = [self._funEval, self._funTime, self._jacEval, self._jacTime]
        data.append(("NO","YES")[self._isFeasible])
        for f in self._ofval:
            data.append(f)
        for (g,r) in zip(self._eqval,self._eqpen):
            data.append(g)
            data.append(r)
        for (g,r) in zip(self._ltval,self._ltpen):
            data.append(g)
            data.append(r)
        for (g,r) in zip(self._gtval,self._gtpen):
            data.append(g)
            data.append(r)
        for (g,r) in zip(self._inval,self._inpen):
            data.append(g)
            data.append(r)
        self._logObj.write(self._logRowFormat.format(*data))
    #end        

    def fun(self,x):
        self._initialize()

        if self._userPreProcessFun:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessFun,shell=True)
        #end

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

        # update the values of the variables
        self._setCurrent(x)

        self._evalAndRetrieveFunctionValues()

        # combine results
        f  = self._ofval.sum()
        f += (self._eqpen*self._eqval**2).sum()
        for (g,r) in zip(self._ltval,self._ltpen): f += r*max(0.0,g)*g
        for (g,r) in zip(self._gtval,self._gtpen): f += r*min(0.0,g)*g
        for (g,r) in zip(self._inval,self._inpen): f += r*(min(0.0,g)+max(1.0,g)-1.0)*g

        self._resetAllValueEvaluations()

        return f
    #end

    def grad(self,x):
        try:
            return self._grad_impl(x)
        except:
            if self._failureMode is "HARD": raise
            return self._old_grad
        #end
    #end

    def _grad_impl(self,x):
        if self._userPreProcessGrad:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessGrad,shell=True)
        #end
        
        os.chdir(os.path.join(self._userDir,self._workDir))
        
        # initializing and updating values was done when evaluating the function

        if self._parallelEval: self._evalJacInParallel()

        # evaluate all required gradients (skip those where the constraint is not active)
        self._jacEval += 1
        self._jacTime -= time.time()

        self._grad[()] = 0.0

        for obj in self._objectives:
            self._grad += obj.function.getGradient(self._variableStartMask)*obj.scale

        for (obj,f,r) in zip(self._constraintsEQ,self._eqval,self._eqpen):
            self._grad += 2.0*r*f*obj.function.getGradient(self._variableStartMask)*obj.scale

        for (obj,f,r) in zip(self._constraintsLT,self._ltval,self._ltpen):
            if f > 0.0:
                self._grad += 2.0*r*f*obj.function.getGradient(self._variableStartMask)*obj.scale

        for (obj,f,r) in zip(self._constraintsGT,self._gtval,self._gtpen):
            if f < 0.0:
                self._grad += 2.0*r*f*obj.function.getGradient(self._variableStartMask)*obj.scale

        for (obj,f,r) in zip(self._constraintsIN,self._inval,self._inpen):
            if f > 1.0 or f < 0.0:
                self._grad += 2.0*r*f*obj.function.getGradient(self._variableStartMask)*obj.scale

        self._grad /= self._varScales

        self._jacTime += time.time()

        # update penalties and params (evaluating the gradient concludes an outer iteration)
        if self._freq > 0:
            if self._jacEval % self._freq is 0: self.update()

        self._resetAllGradientEvaluations()
        os.chdir(self._userDir)

        # make copy to use as fallback
        self._old_grad[()] = self._grad

        return self._grad
    #end

    # if the constraint is active and above tolerance increase the penalty
    # otherwise decrease (minimum and maximum are constrained)
    def update(self,paramsIfFeasible=False):
        self._isFeasible = True
        
        # equality (always active)
        for i in range(self._eqpen.size):
            if abs(self._eqval[i]) > self._tol:
                self._eqpen[i] = min(self._eqpen[i]*self._cup,self._rmax)
                self._isFeasible = False

        # upper bound
        for i in range(self._ltpen.size):
            if self._ltval[i] > self._tol:
                self._ltpen[i] = min(self._ltpen[i]*self._cup,self._rmax)
                self._isFeasible = False
            elif self._ltval[i] < 0.0:
                self._ltpen[i] = max(self._ltpen[i]*self._cdown,self._rini)

        # lower bound
        for i in range(self._gtpen.size):
            if self._gtval[i] < -self._tol:
                self._gtpen[i] = min(self._gtpen[i]*self._cup,self._rmax)
                self._isFeasible = False
            elif self._gtval[i] > 0.0:
                self._gtpen[i] = max(self._gtpen[i]*self._cdown,self._rini)

        # range
        for i in range(self._inpen.size):
            if self._inval[i] > 1.0+self._tol or self._inval[i] < -self._tol:
                self._inpen[i] = min(self._inpen[i]*self._cup,self._rmax)
                self._isFeasible = False
            elif self._inval[i] < 1.0 and self._inval[i] > 0.0:
                self._inpen[i] = max(self._inpen[i]*self._cdown,self._rini)

        # update the values of the parameters
        if not paramsIfFeasible or self._isFeasible:
            for par in self._parameters:
                par.increment()

        # log update
        self._writeLogLine()
    #end

    def feasibleDesign(self):
        return self._isFeasible

