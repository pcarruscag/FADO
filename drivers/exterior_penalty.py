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

import time
import numpy as np
from base_driver import DriverBase


# Exterior Penalty method wrapper
class ExteriorPenaltyDriver(DriverBase):
    def __init__(self, tol, freq=40, rini=8, rmax=1024, factorUp=4, factorDown=0.5):
        # parameters of the method
        self._tol = tol
        self._freq = freq
        self._rini = rini
        self._rmax = rmax
        self._cup = factorUp
        self._cdown = factorDown

        # function values
        self._ofval = None
        self._eqval = None
        self._ltval = None
        self._gtval = None
        self._inval = None

        # constraint penalties
        self._eqpen = None
        self._ltpen = None
        self._gtpen = None
        self._inpen = None

        # timers, counters, flags
        self._funEval = 0
        self._funTime = 0
        self._jacEval = 0
        self._jacTime = 0
        self._isInit = False
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

        self._isInit = True
    #end

    def fun(self,x):
        self._initialize()

        # evaluate everything
        self._funEval += 1
        self._funTime -= time.time()
        
        self._funTime += time.time()

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

        # combine results
        f  = self._ofval.sum()
        f += ((self._eqval*self._eqpen)**2).sum()
        for (g,r) in zip(self._ltval,self._ltpen): f += r*max(0.0,g)*g
        for (g,r) in zip(self._gtval,self._gtpen): f += r*min(0.0,g)*g
        for (g,r) in zip(self._inval,self._inpen): f += r*(min(0.0,g)+max(1.0,g)-1.0)*g

        return f
    #end

    def jac(self,x):
        # evaluate all required gradients (skip those where constraint is not active)
        self._jacEval += 1
        self._jacTime -= time.time()
        
        self._jacTime += time.time()

        # scale gradients as required

        # combine results
        #return df+2*self._r*max(0.0,self._hval)*dh
    #end

    # if the constraint is active and above tolerance increase the penalty
    # otherwise decrease (minimum and maximum are constrained)
    def update(self):
        # equality
        for i in range(self._eqpen.size):
            if abs(self._eqval[i]) > self._tol:
                self._eqpen[i] = min(self._eqpen[i]*self._cup,self._rmax)

        # upper bound
        for i in range(self._ltpen.size):
            if self._ltval[i] > self._tol:
                self._ltpen[i] = min(self._ltpen[i]*self._cup,self._rmax)
            elif self._ltval[i] < 0.0:
                self._ltpen[i] = max(self._ltpen[i]*self._cdown,self._rini)

        # lower bound
        for i in range(self._gtpen.size):
            if self._gtval[i] < -self._tol:
                self._gtpen[i] = min(self._gtpen[i]*self._cup,self._rmax)
            elif self._gtval[i] > 0.0:
                self._gtpen[i] = max(self._gtpen[i]*self._cdown,self._rini)

        # range
        for i in range(self._inpen.size):
            if self._inval[i] > 1.0+self._tol or self._inval[i] < -self._tol:
                self._inpen[i] = min(self._inpen[i]*self._cup,self._rmax)
            elif self._inval[i] < 1.0 and self._inval[i] > 0.0:
                self._inpen[i] = max(self._inpen[i]*self._cdown,self._rini)
    #end

#end


