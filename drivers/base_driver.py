#  Copyright 2019-2020, Pedro Gomes.
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
import numpy as np


# Base class for optimization drivers
# Implements the setup interface
class DriverBase:

    # "structs" to store objective and constraint information
    class _Objective:
        def __init__(self,type,function,scale,weight):
            if scale <= 0.0 or weight <= 0.0:
                raise ValueError("Scale and weight must be positive.")

            if type == "min":
                self.scale = scale*weight
            elif type == "max":
                self.scale = -1.0*scale*weight
            else:
                raise ValueError("Type must be 'min' or 'max'.")

            self.function = function
    #end

    class _Constraint:
        def __init__(self,function,scale,bound1=-1E20,bound2=1E20):
            if scale <= 0.0:
                raise ValueError("Scale must be positive.")

            self.scale = scale
            self.bound1 = bound1
            self.bound2 = bound2
            self.function = function
    #end

    def __init__(self):
        self._variables = []
        self._varScales = None
        self._parameters = []

        # functions by role
        self._objectives = []
        self._constraintsEQ = []
        self._constraintsLT = []
        self._constraintsGT = []
        self._constraintsIN = []

        # function values
        self._ofval = None
        self._eqval = None
        self._ltval = None
        self._gtval = None
        self._inval = None

        # map the start index of each variable in the design vector
        self._variableStartMask = None

        self._userDir = ""
        self._workDir = "__WORKDIR__"
        self._dirPrefix = "DSN_"
        self._keepDesigns = True
        self._failureMode = "HARD"
        self._logObj = None
        self._logColWidth = 13
        self._hisObj = None
        self._hisDelim = ",  "

        self._userPreProcessFun = ""
        self._userPreProcessGrad = ""
    #end

    def addObjective(self,type,function,scale=1.0,weight=1.0):
        self._objectives.append(self._Objective(type,function,scale,weight))

    def addEquality(self,function,target=0.0,scale=1.0):
        self._constraintsEQ.append(self._Constraint(function,scale,target))

    def addUpperBound(self,function,bound=0.0,scale=1.0):
        self._constraintsLT.append(self._Constraint(function,scale,bound))

    def addLowerBound(self,function,bound=0.0,scale=1.0):
        self._constraintsGT.append(self._Constraint(function,scale,bound))

    def addUpLowBound(self,function,lower=-1.0,upper=1.0):
        scale = 1.0/(upper-lower)
        self._constraintsIN.append(self._Constraint(function,scale,lower,upper))

    def setWorkingDirectory(self,dir):
        self._workDir = dir

    def getNumVariables(self):
        N=0
        for var in self._variables: N+=var.getSize()
        return N

    def setLogger(self,obj,width=13):
        self._logObj = obj
        self._logColWidth = width

    def setHistorian(self,obj,delim=",  "):
        self._hisObj = obj
        self._hisDelim = delim

    # methods to retrieve information in a format that the optimizer understands
    def _getConcatenatedVector(self,name):
        x = np.ndarray((self.getNumVariables(),))
        idx = 0
        for var in self._variables:
            for val in var.get(name):
                x[idx] = val
                idx += 1
            #end
        #end
        return x
    #end

    def getInitial(self):
        return self._getConcatenatedVector("Initial")*self._varScales

    def getLowerBound(self):
        return self._getConcatenatedVector("LowerBound")*self._varScales

    def getUpperBound(self):
        return self._getConcatenatedVector("UpperBound")*self._varScales

    # update design variables with the design vector from the optimizer
    def _setCurrent(self,x):
        startIdx = 0
        for var in self._variables:
            endIdx = startIdx+var.getSize()
            var.setCurrent(x[startIdx:endIdx]/var.getScale())
            startIdx = endIdx
        #end
    #end

    def _getVarsAndParsFromFun(self,functions):
        for obj in functions:
            for var in obj.function.getVariables():
                if var not in self._variables: self._variables.append(var)
            for par in obj.function.getParameters():
                if par not in self._parameters: self._parameters.append(par)

            # inform evaluations about which variables they depend on
            for evl in obj.function.getValueEvalChain():
                evl.updateVariables(obj.function.getVariables())
            for evl in obj.function.getGradientEvalChain():
                evl.updateVariables(obj.function.getVariables())
        #end
    #end

    # build variable and parameter vectors from function data
    def preprocessVariables(self):
        # build ordered non duplicated lists of variables and parameters
        self._variables = []
        self._parameters = []
        self._getVarsAndParsFromFun(self._objectives)
        self._getVarsAndParsFromFun(self._constraintsEQ)
        self._getVarsAndParsFromFun(self._constraintsLT)
        self._getVarsAndParsFromFun(self._constraintsGT)
        self._getVarsAndParsFromFun(self._constraintsIN)

        # map the start index of each variable in the design vector
        idx = [0]
        for var in self._variables[0:-1]:
            idx.append(idx[-1]+var.getSize())
        self._variableStartMask = dict(zip(self._variables,idx))

        self._varScales = self._getConcatenatedVector("Scale")

        # store the absolute current path
        self._userDir = os.path.abspath(os.curdir)
    #end

    def setStorageMode(self,keepDesigns=False,dirPrefix="DSN_"):
        self._keepDesigns = keepDesigns
        self._dirPrefix = dirPrefix

    def setFailureMode(self,mode):
        assert mode is "HARD" or mode is "SOFT", "Mode must be either \"HARD\" (exceptions) or \"SOFT\" (default function values)."
        self._failureMode = mode

    def setUserPreProcessFun(self,command):
        self._userPreProcessFun = command

    def setUserPreProcessGrad(self,command):
        self._userPreProcessGrad = command

    def _resetAllValueEvaluations(self):
        for obj in self._objectives:
            obj.function.resetValueEvalChain()
        for obj in self._constraintsEQ:
            obj.function.resetValueEvalChain()
        for obj in self._constraintsLT:
            obj.function.resetValueEvalChain()
        for obj in self._constraintsGT:
            obj.function.resetValueEvalChain()
        for obj in self._constraintsIN:
            obj.function.resetValueEvalChain()
    #end

    def _resetAllGradientEvaluations(self):
        for obj in self._objectives:
            obj.function.resetGradientEvalChain()
        for obj in self._constraintsEQ:
            obj.function.resetGradientEvalChain()
        for obj in self._constraintsLT:
            obj.function.resetGradientEvalChain()
        for obj in self._constraintsGT:
            obj.function.resetGradientEvalChain()
        for obj in self._constraintsIN:
            obj.function.resetGradientEvalChain()
    #end
#end

