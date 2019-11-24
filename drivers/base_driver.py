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


# Base class for optimization drivers
# Implements the setup interface
class DriverBase:

    # "structs" to store objective and constraint information
    class _Objective:
        def __init__(self,type,function,scale,weight):
            if scale <= 0.0 or weight <= 0.0:
                raise ValueError("Scale and weight must be possitive.")

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
                raise ValueError("Scale must be possitive.")

            self.scale = scale
            self.bound1 = bnd1
            self.bound2 = bnd2
            self.function = function
    #end

    def __init__(self):
        self._variables = []
        self._parameters = []
        self._objectives = []
        self._constraintsEQ = []
        self._constraintsLT = []
        self._constraintsGT = []
        self._constraintsIN = []
    #end

    def addVariable(self,variable):
        self._variables.append(variable)

    def addParameter(self,parameter):
        self._parameters.append(parameter)

    def addObjective(self,type,function,scale=1.0,weight=1.0):
        self._objectives.append(self._Objective(type,function,scale,weight))

    def addEquality(self,function,target=0.0,scale=1.0):
        self._constraintsEQ.append(self._Constraint(function,scale,target))

    def addUpperBound(self,function,bound=0.0,scale=1.0):
        self._contraintsLT.append(self._Constraint(function,scale,bound))

    def addLowerBound(self,function,bound=0.0,scale=1.0):
        self._contraintsGT.append(self._Constraint(function,scale,bound))

    def addUpLowBound(self,function,lower=-1.0,upper=1.0,scale=1.0):
        self._constraintIN.append(self._Constraint(function,scale,lower,upper))
#end















