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

from optimizers.line_searches import goldenSection


# Fletcher-Reeves method
def fletcherReeves(fun,x,grad,options,lineSearch=goldenSection):
    # unpack options
    ftol = options["ftol"]
    gtol = options["gtol"]
    maxiter = options["maxiter"]
    verbose = False
    if "disp" in options.keys(): verbose = options["disp"]
    restart = x.size+1
    if "maxcor" in options.keys(): restart = options["maxcor"]
    maxls = 20
    if "maxls" in options.keys(): maxls = options["maxls"]
    tolls = 0.001
    if "tolls" in options.keys(): tolls = options["tolls"]

    # 1D function for line searches
    class lsfun:
        def __init__(self,fun,x,d):
            self._fun = fun
            self._x = x
            self._d = d
        def __call__(self,lbd):
            return self._fun(self._x+lbd*self._d)
    #end

    # initialize
    feval = 1
    jeval = 1
    lbd = 1.0
    f = fun(x)
    G = grad(x)
    success = False

    if verbose:
        headerLine = ""
        for data in ["ITER","FUN EVAL","LS EVAL","STEP","FUN EPS","GRAD EPS","FUN VAL"]:
            headerLine += data.rjust(13)
        logFormat = "{:>13}"*3+"{:>13.6g}"*4
        print("")
    #end

    # start
    for i in range(maxiter):
        # periodic restart
        if i%restart==0 : S=-G

        if verbose and i%10==0 : print(headerLine)
        
        # line search
        f_old = f
        (lbd,f,nls) = lineSearch(lsfun(fun,x,S),maxls,f,lbd,tolls)
        feval += nls

        # detect bad direction and restart
        if f>f_old:
            if verbose: print("Bad search direction, taking steepest descent.")
            f = f_old
            S = -G
            (lbd,f,nls2) = lineSearch(lsfun(fun,x,S),maxls,f,lbd,tolls)
            nls += nls2
            feval += nls
            if f>f_old:
                f = f_old
                if verbose: print("Could not improve design further.")
                break
        #end

        # update search direction
        x += lbd*S
        G_old = G
        S_old = S
        G = grad(x)
        jeval += 1
        S = -G+G.dot(G)/G_old.dot(G_old)*S_old

        # log
        logData = [i+1, feval, nls, lbd, f_old-f, max(abs(G)), f]
        if verbose: print(logFormat.format(*logData))

        # convergence criteria
        if f_old-f < ftol or max(abs(G)) < gtol:
            success = True
            break
    #end

    result = {"x" : x, "fun" : f, "jac" : G, "nit" : i+1,
              "nfev" : feval, "njev" : jeval, "success" : success}
    return result
#end
