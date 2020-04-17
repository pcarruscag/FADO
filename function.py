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
import abc

# Abstract base class for functions
class FunctionBase(abc.ABC):
    def __init__(self,name):
        self._name = name
        # inputs
        self._variables = []

    def getName(self,maxLen=0):
        name = self._name
        if maxLen==0: return name
        if maxLen<len(name): name = name[:maxLen]
        return name

    def getVariables(self):
        return self._variables

    @abc.abstractmethod
    def getParameters(self):
        return NotImplemented

    @abc.abstractmethod
    def getValue(self):
        return NotImplemented

    @abc.abstractmethod
    def getGradient(self,mask):
        return NotImplemented

    @abc.abstractmethod
    def resetValueEvalChain(self):
        return NotImplemented

    @abc.abstractmethod
    def resetGradientEvalChain(self):
        return NotImplemented

    @abc.abstractmethod
    def getValueEvalChain(self):
        return NotImplemented

    @abc.abstractmethod
    def getGradientEvalChain(self):
        return NotImplemented
#end


# Class to define evaluation-based functions
class Function(FunctionBase):
    def __init__(self,name="",outFile="",outParser=None):
        FunctionBase.__init__(self,name)

        # where and how the output value is obtained
        self.setOutput(outFile,outParser)

        # evaluation pipelines for value and gradient
        self._funEval = []
        self._gradEval = []

        # where and how their gradients are obtained
        self._gradFiles = []
        self._gradParse = []

    def addInputVariable(self,variable,gradFile,gradParser):
        self._variables.append(variable)
        self._gradFiles.append(gradFile)
        self._gradParse.append(gradParser)

    def getParameters(self):
        parameters = []
        for evl in self._funEval:
            parameters += evl.getParameters()
        for evl in self._gradEval:
            parameters += evl.getParameters()
        return parameters

    def setOutput(self,file,parser):
        self._outFile = file
        self._outParser = parser

    def addValueEvalStep(self,evaluation):
        self._funEval.append(evaluation)

    def addGradientEvalStep(self,evaluation):
        self._gradEval.append(evaluation)

    def getValue(self):
        # check if we can retrive the value
        for evl in self._funEval:
            if not evl.isRun():
                self._sequentialEval(self._funEval)
                break
        #end
        return self._outParser.read(self._outFile)

    def getGradient(self,mask=None):
        # check if we can retrive the gradient
        for evl in self._gradEval:
            if not evl.isRun():
                self._sequentialEval(self._gradEval)
                break
        #end

        # determine size of gradient vector
        size = 0
        if mask is None: src = self._variables
        else:            src = mask.keys()
        for var in src:
            size += var.getSize()

        # populate gradient vector
        gradient = np.ndarray((size,))
        idx = 0
        for var,file,parser in zip(self._variables,self._gradFiles,self._gradParse):
            grad = parser.read(file)
            if var.getSize() == 1:
                try: grad = sum(grad)
                except: pass
            #end
            if mask is not None: idx = mask[var]
            try:
                for val in grad:
                    gradient[idx] = val
                    idx += 1
            except:
                gradient[idx] = grad
                idx += 1
            #end
        #end
        
        return gradient
    #end

    def _sequentialEval(self,evals):
        for evl in evals:
            evl.initialize()
            evl.run()
        #end
    #end

    def resetValueEvalChain(self):
        self._resetEvals(self._funEval)

    def resetGradientEvalChain(self):
        self._resetEvals(self._gradEval)

    def _resetEvals(self,evals):
        for evl in evals:
            evl.finalize()
        #end
    #end

    def getValueEvalChain(self):
        return self._funEval

    def getGradientEvalChain(self):
        return self._gradEval
#end


# Continuous measure of non-dscreteness (usually to use as a constraint)
class NonDiscreteness(FunctionBase):
    def __init__(self,name=""):
        FunctionBase.__init__(self,name)

    def addInputVariable(self,variable):
        self._variables.append(variable)

    def getValue(self):
        y = 0.0
        N = 0
        for var in self._variables:
            N += var.getSize()
            x  = var.getCurrent()
            lb = var.getLowerBound()
            ub = var.getUpperBound()
            y += ((ub-x)*(x-lb)/(ub+lb)**2).sum()
        return 4*y/N

    def getGradient(self,mask=None):
        # determine size of gradient vector
        N = 0
        for var in self._variables:
            N += var.getSize()

        size = 0
        if mask is None:
            size = N
        else:
            for var in mask.keys():
                size += var.getSize()

        # populate gradient vector
        gradient = np.ndarray((size,))
        idx = 0
        for var in self._variables:
            x  = var.getCurrent()
            lb = var.getLowerBound()
            ub = var.getUpperBound()
            grad = (4.0/N)*(ub+lb-2*x)/(ub+lb)**2

            if mask is not None: idx = mask[var]

            try:
                for val in grad:
                    gradient[idx] = val
                    idx += 1
            except:
                gradient[idx] = grad
                idx += 1
            #end
        #end

        return gradient
    #end

    def getParameters(self):
        return []

    def resetValueEvalChain(self):
        pass

    def resetGradientEvalChain(self):
        pass

    def getValueEvalChain(self):
        return []

    def getGradientEvalChain(self):
        return []
#end

