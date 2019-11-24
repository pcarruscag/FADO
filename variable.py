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

import copy
import numpy as np


# Class for design variables
class InputVariable:
    # parser specifies how the variable is written to file
    # size >= 1 defines a vector variable whose x0, lb, and ub values are broadcast
    # size == 0 means auto, size determined from x0, lb/ub must be compatible or scalar
    def __init__(self, x0, parser, size=0, lb=-1E20, ub=1E20):
        self._parser = parser

        if size == 0 and isinstance(x0,float): size=1
        if size >= 1:
            try:
                assert(isinstance(x0,float))
                assert(isinstance(lb,float))
                assert(isinstance(ub,float))
            except:
                raise ValueError("If size is specified, x0, lb, and ub must be scalars.")
            #end
            self._x0 = np.ones((size,))*x0
            self._lb = np.ones((size,))*lb
            self._ub = np.ones((size,))*ub
        else:
            try:
                size = x0.size
                assert(size>=1)
                self._x0 = x0
                if not isinstance(lb,float):
                    assert(lb.size == size)
                    self._lb = lb
                else:
                    self._lb = np.ones((size,))*lb
                #end
                if not isinstance(ub,float):
                    assert(ub.size == size)
                    self._ub = ub
                else:
                    self._ub = np.ones((size,))*ub
                #end
            except:
                raise ValueError("Incompatible sizes of x0, lb, and ub.")
            #end
        #end

        self._size = size
        self._x = copy.deepcopy(self._x0)
    #end

    def getSize(self):
        return self._size

    def getInitial(self):
        return self._x0

    def getCurrent(self):
        return self._x

    def getLowerBound(self):
        return self._lb

    def getUpperBound(self):
        return self._ub

    def setCurrent(self,x):
        for i in range(x.size): self._x[i] = x[i]

    def writeToFile(self,file):
        self._parser.write(file,self._x)
#end


# Class for parameters
class Parameter:
    # values is an indexable structure (e.g. range, list)
    # function can be used to further convert the current value
    def __init__(self,values,parser,start=0,function=None):
        self._values = values
        self._parser = parser
        self._function = function
        # make sure starting possition is valid
        self._upper = len(values)-1
        self._index = max(0,min(self._upper,start))

    # inc/dec-menting returns true at the bounds
    def increment(self):
        self._index = max(0,min(self._upper,self._index+1))
        return (self._index == self._upper)

    def decrement(self):
        self._index = max(0,min(self._upper,self._index-1))
        return (self._index == 0)

    def writeToFile(self,file):
        value = self._values[self._index]
        if self._function != None:
            value = self._function(value)
        self._parser.write(file,value)
#end
