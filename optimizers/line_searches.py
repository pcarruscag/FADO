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


# 1D minimization using the Golden Section method
def goldenSection(fun,maxiter,f0=None,lbd0=1,tol=1e-3):
    # look for an interval that contains the minimum
    # assuming we have a descent direction
    feval = 0
    x = [0.0, 0.0, lbd0]
    if f0 is None:
        f0 = fun(0.0)
        feval += 1
    y = [f0,  f0, fun(lbd0)]
    feval += 1

    # while the function decreases the step is doubled
    while y[2]<y[1]:
        x[1] = x[2]
        y[1] = y[2]
        lbd0 *= 2.0
        x[2] = x[1]+lbd0
        y[2] = fun(x[2])
        feval += 1
    #end

    # initial points
    L2 = 0.381966*x[2] # 2/(3+sqrt(5))*x2
    x = [0.0, L2, x[2]-L2, x[2]]
    y = [fun(x[1]), fun(x[2])]
    feval += 2
    
    # iterate
    while feval < maxiter:
        if y[0] < y[1]: # keep left interval
            x[3] = x[2]
            x[2] = x[1]
            # new test point
            x[1] = x[0]+(x[3]-x[2])
            y[1] = y[0]
            y[0] = fun(x[1])
            feval += 1
        else:           # keep right interval
            x[0] = x[1]
            x[1] = x[2]
            # new test point
            x[2] = x[3]-(x[1]-x[0])
            y[0] = y[1]
            y[1] = fun(x[2])
            feval += 1
        #end

        # check convergence
        if abs(x[2]-x[1])/L2 < tol:
            break
    #end

    # minimum value found
    if y[0] < y[1]:
        x_opt = x[1]
        y_min = y[0]
    else:
        x_opt = x[2]
        y_min = y[1]
    #end

    return (x_opt,y_min,feval)
#end
