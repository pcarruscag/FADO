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

import numpy as np


# Create a parameterization that respects bound constraints on variables
# x=0 : x'=lb, x=pi/2 : x'=ub
class BoundConstraints:
    def __init__(self,fun,grad,lb,ub):
        self._fun = fun
        self._grad = grad
        self._lb = lb
        self._range = ub-lb
        self._partials = None
        self._y = None
    #end

    def __call__(self,x):
        self._y = 0.5*np.pi*(x-self._lb)/self._range
        self._partials = 0.5*np.pi*np.sin(2.0*self._y)
        return self._lb+self._range*np.sin(self._y)**2.0
    #end

    def inverse(self,x):
        return np.arcsin(np.sqrt((x-self._lb)/self._range))*2.0/np.pi*self._range+self._lb
    #end

    def fun(self,x):
        return self._fun(self(x))
    #end

    def grad(self,x):
        g = self._grad(self(x))
        return g*self._partials
    #end
#end


# Scale gradient without scaling variables
class GradientScale:
    def __init__(self,grad,scale):
        self._grad = grad
        self._scale = scale

    def grad(self,x):
        return self._grad(x)*self._scale
#end
