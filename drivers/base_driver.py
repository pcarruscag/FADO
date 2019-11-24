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
        self._varScales = []
        self._parameters = []
        self._objectives = []
        self._constraintsEQ = []
        self._constraintsLT = []
        self._constraintsGT = []
        self._constraintsIN = []
        self.__workDir = "__WORKDIR__"
    #end

    def addVariable(self,variable,scale=1.0):
        self._variables.append(variable)
        self._varScales.append(scale)

    def addParameter(self,parameter):
        self._parameters.append(parameter)

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

    def getNumVariables(self):
        N=0
        for var in self._variables: N+=var.getSize()
        return N

    # methods to retrieve information in a format that the optimizer understands
    def _getConcatenatedVector(self,getterName):
        x = np.ndarray((self.getNumVariables(),))
        idx = 0
        for var in self._variables:
            for val in getattr(var,getterName)():
                x[idx] = val
                idx += 1
            #end
        #end
        return x
    #end
    
    def getInitial(self):
        return self._getConcatenatedVector("getInitial")

    def getLowerBound(self):
        return self._getConcatenatedVector("getLowerBound")

    def getUpperBound(self):
        return self._getConcatenatedVector("getUpperBound")

    # update design variables with design vector from the optimizer
    def setCurrent(self,x):
        startIdx = 0
        for (var,scale) in zip(self._variables,self._varScales):
            endIdx = startIdx+var.getSize()
            var.setCurrent(x[startIdx:endIdx]/scale)
            startIdx = endIdx
        #end
    #end

#end















